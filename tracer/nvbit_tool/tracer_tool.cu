/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bitset>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

/* zstd library for direct .traceg.zst output */
#include <zstd.h>
#include <cmath>
#include <tuple>

#define TRACER_VERSION "5"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
int lineinfo = 0;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

/* Use xz to compress the *.trace file */
int xz_compress_trace = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

std::string user_folder = getcwd(NULL, 0);
std::string cwd = getcwd(NULL, 0);
std::string traces_location = cwd + "/traces/";
std::string kernelslist_location = cwd + "/traces/kernelslist";
std::string stats_location = cwd + "/traces/stats.csv";

std::unordered_map<CUcontext, std::string> ctx_kernelslist;
std::unordered_map<CUcontext, std::string> ctx_stats_location;
std::unordered_map<CUcontext, int> ctx_kernelid;
std::unordered_map<CUcontext, std::string> ctx_current_kernel_name;

/* Per-kernel header info for direct .zsim output */
struct KernelHeader {
  std::string kernel_name;
  int kernel_id;
  unsigned gridDimX, gridDimY, gridDimZ;
  unsigned blockDimX, blockDimY, blockDimZ;
  unsigned shmem;
  int nregs;
  int binary_version;
  uint64_t cuda_stream_id;
  uint64_t shmem_base_addr;
  uint64_t local_mem_base_addr;
};

/* Per-context kernel header and TB instruction buffers */
std::unordered_map<CUcontext, KernelHeader> ctx_kernel_header;
/* Per-context TB buffers: map from (cta_x, cta_y, cta_z) to per-warp
 * instruction vectors */
typedef std::map<std::tuple<int, int, int>,
                 std::vector<std::vector<std::string>>>
    TBBufferMap;
std::unordered_map<CUcontext, TBBufferMap> ctx_tb_buffers;
/* Per-context LDGSTS flags (same dedup as post-processing) */
std::unordered_map<CUcontext,
                   std::map<std::tuple<int, int, int, int>, bool>>
    ctx_ldgsts_flags;

std::string kernel_ranges = "";

void trim_string(std::string &str);

/* gpubench P1b live sampling: mode word baked into the injected call.
 * Parsed from PTRACE_SAMPLE in nvbit_at_init.
 *
 * The parse result is staged in a plain host struct and published to the
 * managed ptrace_sample_list lazily, at the first kernel launch, while the
 * GPU is idle after cudaDeviceSynchronize. Touching __managed__ memory from
 * nvbit_at_init (no CUDA context yet) or from inside NVBit's context-init
 * callback wedges NVBit's driver interception — both observed on this
 * driver (no events fire / first instrumented launch deadlocks). */
uint32_t ptrace_sample_mode = 0;
std::string ptrace_sample_spec = "";
ptrace_sample_list_t ptrace_sample_host;
ptrace_sample_list_t *ptrace_sample_list_dev = NULL; // device copy (cudaMalloc)

/* Relaunch dedup (PTRACE_DEDUP=K): trace only the first K launches of each
 * distinct (kernel name, grid dim, block dim, shmem) signature; later
 * relaunches run uninstrumented and their kernelslist entry references the
 * first instance's trace file, so the replay order stays complete. Iterative
 * benchmarks (jacobi, fdtd, graph frontiers) relaunch identical kernels
 * hundreds of times — this removes the dominant collection multiplier.
 * Caveat: ping-pong kernels alternate buffer pointers between launches; the
 * deduped entries reuse the first instance's addresses (strides and footprint
 * are unaffected, absolute bases are not). 0/unset = off. */
int ptrace_dedup_limit = 0;
std::unordered_map<std::string, int> ptrace_dedup_count;
std::unordered_map<std::string, std::string> ptrace_dedup_first_file;
uint64_t ptrace_dedup_skipped = 0;

/* Parse PTRACE_SAMPLE. Grammar (semicolon-separated tokens):
 *   ""            → disabled, trace every block (default)
 *   "corner"      → record blocks with ctaid.{x,y,z} < 2
 *   "x,y,z"       → record this block (repeatable)
 *   "corner;4,0,0"→ corner sample plus explicit extras
 * PTRACE_SAMPLE_COUNT_SKIPPED=1 additionally counts skipped blocks'
 * warp instructions in the total counter (slower). */
void parse_ptrace_sample_from_env() {
  ptrace_sample_mode = 0;
  ptrace_sample_host.n = 0;

  const char *env = std::getenv("PTRACE_SAMPLE");
  if (!env || std::string(env).empty())
    return;

  std::string spec(env);
  ptrace_sample_spec = spec;
  std::istringstream stream(spec);
  std::string token;
  while (std::getline(stream, token, ';')) {
    trim_string(token);
    if (token.empty())
      continue;
    if (token == "count") {
      /* Counting-only profile: no block matches the (empty) explicit list,
       * every warp instruction increments the total counter. stats.csv then
       * carries per-kernel warp-inst counts — the measured side of the
       * generation gate when ncu/perf counters are unavailable. */
      ptrace_sample_mode |= PTRACE_SAMPLE_LIST | PTRACE_SAMPLE_COUNT_SKIPPED;
      continue;
    }
    if (token == "corner" || token.rfind("corner:", 0) == 0) {
      ptrace_sample_mode |= PTRACE_SAMPLE_CORNER;
      if (token.rfind("corner:", 0) == 0) {
        int k = atoi(token.c_str() + 7);
        if (k < 2 || k > 255) {
          fprintf(stderr, "PTRACE_SAMPLE: corner width %d out of range "
                          "[2,255], using 2\n",
                  k);
        } else {
          ptrace_sample_mode |= ((uint32_t)k << 8);
        }
      }
      continue;
    }
    int x, y, z;
    if (sscanf(token.c_str(), "%d,%d,%d", &x, &y, &z) == 3) {
      if (ptrace_sample_host.n >= PTRACE_MAX_SAMPLE_BLOCKS) {
        fprintf(stderr,
                "PTRACE_SAMPLE: too many explicit blocks (max %d), "
                "ignoring %s\n",
                PTRACE_MAX_SAMPLE_BLOCKS, token.c_str());
        continue;
      }
      ptrace_sample_host.x[ptrace_sample_host.n] = x;
      ptrace_sample_host.y[ptrace_sample_host.n] = y;
      ptrace_sample_host.z[ptrace_sample_host.n] = z;
      ptrace_sample_host.n++;
      ptrace_sample_mode |= PTRACE_SAMPLE_LIST;
    } else {
      fprintf(stderr, "PTRACE_SAMPLE: cannot parse token '%s'\n",
              token.c_str());
    }
  }

  const char *count_skipped = std::getenv("PTRACE_SAMPLE_COUNT_SKIPPED");
  if (ptrace_sample_mode != 0 && count_skipped &&
      std::string(count_skipped) == "1") {
    ptrace_sample_mode |= PTRACE_SAMPLE_COUNT_SKIPPED;
  }

  printf("PTRACE_SAMPLE: mode=0x%x spec='%s' explicit_blocks=%d\n",
         ptrace_sample_mode, spec.c_str(), ptrace_sample_host.n);
}

void parse_ptrace_dedup_from_env() {
  const char *env = std::getenv("PTRACE_DEDUP");
  if (!env || std::string(env).empty())
    return;
  ptrace_dedup_limit = atoi(env);
  if (ptrace_dedup_limit < 0)
    ptrace_dedup_limit = 0;
  if (ptrace_dedup_limit > 0) {
    printf("PTRACE_DEDUP: tracing first %d launch(es) per kernel signature\n",
           ptrace_dedup_limit);
  }
}

struct KernelRange {
  uint64_t start;
  uint64_t end; // UINT64_MAX means open-ended
  std::vector<std::regex>
      kernel_name_regexes; // Vector of regexes for multiple patterns
};
std::vector<KernelRange> g_kernel_ranges;
uint64_t g_max_kernel_id = 0;
void parse_kernel_ranges_from_env() {
  g_kernel_ranges.clear();
  g_max_kernel_id = 0;

  const char *env_var = std::getenv("DYNAMIC_KERNEL_RANGE");
  if (!env_var || std::string(env_var).empty()) {
    g_kernel_ranges.push_back({0, 0, {std::regex(".*")}}); // 0 end = trace all
    return;
  }
  std::string input(env_var);
  std::istringstream stream(input);
  std::string token;

  while (stream >> token) {
    if (token.empty())
      continue;

    uint64_t start = 0, end = 0;
    std::vector<std::regex> regexes;

    size_t at_pos = token.find('@');
    std::string range_part, regex_part;

    if (at_pos != std::string::npos) {
      range_part = token.substr(0, at_pos);
      regex_part = token.substr(at_pos + 1);
    } else {
      range_part = token;
    }

    // Parse the range
    if (!range_part.empty()) {
      size_t dash_pos = range_part.find('-');
      if (dash_pos != std::string::npos) {
        std::string start_str = range_part.substr(0, dash_pos);
        std::string end_str = range_part.substr(dash_pos + 1);

        start = std::stoull(start_str);
        if (!end_str.empty()) {
          end = std::stoull(end_str);
        } else {
          end = 0; // open-ended
        }
      } else {
        start = std::stoull(range_part);
        end = start;
      }
    } else {
      // No range → match all IDs
      start = 0;
      end = 0;
    }

    // Parse the regexes
    if (!regex_part.empty()) {
      std::istringstream regex_stream(regex_part);
      std::string regex_token;
      while (std::getline(regex_stream, regex_token, ',')) {
        try {
          regexes.emplace_back(regex_token);
        } catch (const std::regex_error &e) {
          std::cerr << "Invalid regex: " << regex_token << std::endl;
        }
      }
    } else {
      regexes.emplace_back(".*"); // match all kernel names
    }

    g_kernel_ranges.push_back({start, end, regexes});
    if (end > g_max_kernel_id) {
      g_max_kernel_id = end;
    }
  }
}

bool should_trace_kernel(uint64_t kernel_id, const std::string &kernel_name) {
  for (const auto &range : g_kernel_ranges) {
    // Check range for kernel ID
    if (range.end == 0) {
      if (kernel_id >= range.start) {
        // Match any of the regexes for this range
        for (const auto &regex : range.kernel_name_regexes) {
          if (std::regex_match(kernel_name, regex)) {
            return true;
          }
        }
      }
    } else if (kernel_id >= range.start && kernel_id <= range.end) {
      // Match any of the regexes for this range
      for (const auto &regex : range.kernel_name_regexes) {
        if (std::regex_match(kernel_name, regex)) {
          return true;
        }
      }
    }
  }
  return false;
}

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

/* File pointers for the kernels, and stats files */
static FILE *kernelsFile = NULL;
static FILE *statsFile = NULL;
static bool first_call = true;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;

/* Spinlock fast forward control */
int enable_spinlock_fast_forward = 0;
int spinlock_iter_to_keep = 0;
// Map from kernel name to spinlock instruction indices
std::map<std::string, std::vector<uint32_t> *> spinlock_instr_map;
std::pair<std::string, std::vector<uint32_t>>
parse_spinlock_instructions(const std::string &line);

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(lineinfo, "TRACE_LINEINFO", 0,
              "Include source code line info at the start of each traced line. "
              "The target binary must be compiled with -lineinfo or "
              "--generate-line-info");
  GET_VAR_STR(
      kernel_ranges, "DYNAMIC_KERNEL_RANGE",
      "Specify kernel IDs or ranges to trace. Format:\n"
      "  - Single ID:       \"2\" traces only kernel 2.\n"
      "  - Range:           \"5-8\" traces kernels 5 through 8 (inclusive).\n"
      "  - Open-ended:      \"10-\" traces from kernel 10 onward.\n"
      "  - Multiple ranges: \"2 5-8 10-\" (space-separated).\n"
      "  - With regex:      \"5-8@kernel_a.*,kernel_b.*\" traces kernels 5–8 "
      "with matching names.\n"
      "If unset or empty, all kernels will be traced from the beginning.");
  GET_VAR_INT(
      active_from_start, "ACTIVE_FROM_START", 1,
      "Start instruction tracing from start or wait for cuProfilerStart "
      "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_RANGE options have no "
      "effect");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(enable_compress, "TOOL_COMPRESS", 1, "Enable traces compression");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(
      terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT",
      0, "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0,
              "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  GET_VAR_INT(xz_compress_trace, "TRACE_FILE_COMPRESS", 1,
              "Create xz-compressed trace"
              "file");
  GET_VAR_INT(enable_spinlock_fast_forward, "ENABLE_SPINLOCK_FAST_FORWARD", 0,
              "Enable spinlock fast forwarding");
  GET_VAR_INT(spinlock_iter_to_keep, "SPINLOCK_ITER_TO_KEEP", 1,
              "Number of iterations to keep for spinlock fast forwarding");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  active_region = false;
  char *usr_defined_folder = std::getenv("TRACES_FOLDER");
  if (usr_defined_folder != NULL)
    user_folder = usr_defined_folder;
  parse_kernel_ranges_from_env();
  parse_ptrace_sample_from_env();
  parse_ptrace_dedup_from_env();

  // Read in the spinlock_instructions.txt and build a map from kernel name to
  // spinlock instruction indices
  if (enable_spinlock_fast_forward) {
    std::string spinlock_instr_file =
        user_folder + "/spinlock_detection/spinlock_instructions.txt";
    std::ifstream instr_fs(spinlock_instr_file);
    std::string line;
    while (std::getline(instr_fs, line)) {
      auto [kernel_name, indices] = parse_spinlock_instructions(line);
      spinlock_instr_map[kernel_name] = new std::vector<uint32_t>(indices);
    }
    instr_fs.close();
  }
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose >= 1) {
      printf("Inspecting function %s at address 0x%lx\n",
             nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
      uint32_t line_num = 0;
      // Temporary workaround for a bug in NVBit 1.7.4, which does not correctly
      // handle `call.rel`. Instrumenting this instruction leads to illegal
      // memory access. Refer to:
      // https://github.com/NVlabs/NVBit/issues/142#issue-2911561744
      if (!strcmp(instr->getOpcode(), "CALL.REL.NOINC")) {
        printf("Warning: Ignoring CALL.REL.NOINC (NVBit 1.7.4 bug)\n");
        continue;
      }

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      if (verbose >= 2) {
        instr->printDecoded();
      }

      if (lineinfo) {
        char *file_name, *dir_name;
        nvbit_get_line_info(ctx, func, instr->getOffset(), &file_name,
                            &dir_name, &line_num);
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];

      /* check all operands. For now, we ignore constant, TEX, predicates and
       * unified registers. We only report vector regisers */
      int src_oprd[MAX_SRC];
      int srcNum = 0;
      int dst_oprd = -1;
      int mem_oper_idx = -1;
      int num_mref = 0;
      uint64_t imm_value = 0;

      for (int i = 0; i < instr->getNumOperands(); ++i) {
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF) {
          assert(srcNum < MAX_SRC);
          src_oprd[srcNum] = instr->getOperand(i)->u.mref.ra_num;
          srcNum++;
          mem_oper_idx++;
          num_mref++;
          // if(mem_oper_idx == 0){
          //   mem_oper_idx = 1; // loop control
          // }
        } else if (op->type == InstrType::OperandType::REG) {
          if (i == 0) {
            // find dst reg
            dst_oprd = instr->getOperand(0)->u.reg.num;
          } else {
            // find src regs
            assert(srcNum < MAX_SRC);
            src_oprd[srcNum] = instr->getOperand(i)->u.reg.num;
            srcNum++;
          }
        }
        // Add immediate value for DEPBAR instruction
        else if (op->type == InstrType::OperandType::IMM_UINT64) {
          imm_value = instr->getOperand(i)->u.imm_uint64.value;
        }
      }

      do {
        /* insert call to the instrumentation function with its
         * arguments */
        nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);

        /* pass predicate value */
        nvbit_add_call_arg_guard_pred_val(instr);

        /* send opcode and pc */
        nvbit_add_call_arg_const_val32(instr, opcode_id);
        nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());

        /* mem addresses info */
        if (mem_oper_idx >= 0) {
          nvbit_add_call_arg_const_val32(instr, 1);
          assert(num_mref <= 2);
          if (num_mref == 2) { // LDGSTS
            nvbit_add_call_arg_mref_addr64(instr, 1 - mem_oper_idx);
          } else {
            nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
          }
          nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
        } else {
          nvbit_add_call_arg_const_val32(instr, 0);
          nvbit_add_call_arg_const_val64(instr, static_cast<uint64_t>(-1));
          nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(-1));
        }

        /* reg info */
        nvbit_add_call_arg_const_val32(instr, dst_oprd);
        for (int i = 0; i < srcNum; i++) {
          nvbit_add_call_arg_const_val32(instr, src_oprd[i]);
        }
        for (int i = srcNum; i < MAX_SRC; i++) {
          nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(-1));
        }
        nvbit_add_call_arg_const_val32(instr, srcNum);

        /* immediate info */
        nvbit_add_call_arg_const_val64(instr, imm_value);

        /* add pointer to channel_dev and the consolidated control block
         * (one pointer instead of three — see arg-count budget note in
         * inject_funcs.cu) */
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&tool_ctrl);
        /* gpubench P1b live sampling: mode word + explicit block list
         * (device pointer; cudaMalloc'd at first launch, before any
         * instrumentation runs — NULL when list mode is unused). The 64-bit
         * pointer precedes the 32-bit mode word to keep it even-aligned in
         * NVBit's arg marshaling (see inject_funcs.cu). */
        nvbit_add_call_arg_const_val64(instr,
                                       (uint64_t)ptrace_sample_list_dev);
        nvbit_add_call_arg_const_val32(instr, ptrace_sample_mode);
        /* Add Source code line number for current instr */
        nvbit_add_call_arg_const_val32(instr, (int)line_num);
        /* Add instruction index for current instr (spinlock detection) */
        nvbit_add_call_arg_const_val32(instr, (uint32_t)instr->getIdx());
        mem_oper_idx--;
      } while (mem_oper_idx >= 0);

      cnt++;
    }
  }
}

__global__ void flush_channel() {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  channel_dev.push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  channel_dev.flush();
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                                nvbit_api_cuda_t cbid, void *params,
                                bool stream_capture = false,
                                bool build_graph = false) {
  // no need to sync during stream capture or manual graph build, since no
  // kernel is actually launched.
  if (!stream_capture && !build_graph) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
  }

  /* Publish the staged sample plan to device memory on the first launch,
   * while the GPU is idle and before any instrumentation bakes the pointer
   * (see comment at ptrace_sample_host). */
  if (ptrace_sample_list_dev == NULL &&
      (ptrace_sample_mode & PTRACE_SAMPLE_LIST) && !stream_capture &&
      !build_graph) {
    CUDA_SAFECALL(
        cudaMalloc(&ptrace_sample_list_dev, sizeof(ptrace_sample_list_t)));
    CUDA_SAFECALL(cudaMemcpy(ptrace_sample_list_dev, &ptrace_sample_host,
                             sizeof(ptrace_sample_list_t),
                             cudaMemcpyHostToDevice));
    printf("PTRACE_SAMPLE: device list at %p (n=%d)\n",
           (void *)ptrace_sample_list_dev, ptrace_sample_host.n);
  }

  // Mark if the kernel should be traced
  std::string func_name = std::string(nvbit_get_func_name(ctx, func, true));
  if (active_from_start && should_trace_kernel(ctx_kernelid[ctx], func_name))
    active_region = true;

  // Terminate tracing if the limit number of kernels is reached
  if (terminate_after_limit_number_of_kernels_reached && g_max_kernel_id != 0 &&
      ctx_kernelid[ctx] > g_max_kernel_id) {
    exit(0);
  }

  // Get launch config for this kernel
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
      cbid == API_CUDA_cuLaunchKernelEx) {
    cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
    gridDimX = p->config->gridDimX;
    gridDimY = p->config->gridDimY;
    gridDimZ = p->config->gridDimZ;
    blockDimX = p->config->blockDimX;
    blockDimY = p->config->blockDimY;
    blockDimZ = p->config->blockDimZ;
    sharedMemBytes = p->config->sharedMemBytes;
    hStream = p->config->hStream;
  } else {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
    gridDimX = p->gridDimX;
    gridDimY = p->gridDimY;
    gridDimZ = p->gridDimZ;
    blockDimX = p->blockDimX;
    blockDimY = p->blockDimY;
    blockDimZ = p->blockDimZ;
    sharedMemBytes = p->sharedMemBytes;
    hStream = p->hStream;
  }

  // Get the number of registers and shared memory size for the kernel
  int nregs;
  CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

  int shmem_static_nbytes;
  CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

  int binary_version;
  CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                                   CU_FUNC_ATTRIBUTE_BINARY_VERSION, func));

  // Instrument the kernel if needed
  instrument_function_if_needed(ctx, func);

  // Relaunch dedup: skip instrumentation for repeat launches of an
  // already-traced (name, geometry, shmem) signature.
  std::string dedup_sig;
  bool dedup_skip = false;
  if (ptrace_dedup_limit > 0 && active_region) {
    char sig_buf[2048];
    snprintf(sig_buf, sizeof(sig_buf), "%p|%s|%u,%u,%u|%u,%u,%u|%u",
             (void *)ctx, func_name.c_str(), gridDimX, gridDimY, gridDimZ,
             blockDimX, blockDimY, blockDimZ,
             (unsigned)(shmem_static_nbytes + sharedMemBytes));
    dedup_sig = sig_buf;
    int &cnt = ptrace_dedup_count[dedup_sig];
    if (cnt >= ptrace_dedup_limit) {
      dedup_skip = true;
      ptrace_dedup_skipped++;
    } else {
      cnt++;
    }
  }

  // Enable or disable tracing based on the active region
  if (active_region && !dedup_skip) {
    nvbit_enable_instrumented(ctx, func, true);
    tool_ctrl.stop = false;
  } else {
    nvbit_enable_instrumented(ctx, func, false);
    tool_ctrl.stop = true;
  }

  /* Store header info for later .zsim generation (no file opened here). */
  char buffer[2048];

  if (!tool_ctrl.stop) {
    KernelHeader &hdr = ctx_kernel_header[ctx];
    hdr.kernel_name = std::string(nvbit_get_func_name(ctx, func, true));
    hdr.kernel_id = ctx_kernelid[ctx];
    hdr.gridDimX = gridDimX;
    hdr.gridDimY = gridDimY;
    hdr.gridDimZ = gridDimZ;
    hdr.blockDimX = blockDimX;
    hdr.blockDimY = blockDimY;
    hdr.blockDimZ = blockDimZ;
    hdr.shmem = shmem_static_nbytes + sharedMemBytes;
    hdr.nregs = nregs;
    hdr.binary_version = binary_version;
    hdr.cuda_stream_id = (uint64_t)hStream;
    hdr.shmem_base_addr = (uint64_t)nvbit_get_shmem_base_addr(ctx);
    hdr.local_mem_base_addr = (uint64_t)nvbit_get_local_mem_base_addr(ctx);

    /* Initialize buffers for this kernel. */
    ctx_tb_buffers[ctx].clear();
    ctx_ldgsts_flags[ctx].clear();

    printf("Tracing kernel %d to .zsim (in-memory)\n", ctx_kernelid[ctx]);
  }

  /* Write kernelslist entry with .zsim suffix. */
  kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
  sprintf(buffer, "kernel-%d-ctx_0x%lx.trace.zsim", ctx_kernelid[ctx],
          (unsigned long)ctx);
  if (!tool_ctrl.stop) {
    fprintf(kernelsFile, "%s", buffer);
    fprintf(kernelsFile, "\n");
    /* Remember this signature's trace file for later deduped relaunches. */
    if (ptrace_dedup_limit > 0 && !dedup_sig.empty() &&
        ptrace_dedup_first_file.find(dedup_sig) ==
            ptrace_dedup_first_file.end()) {
      ptrace_dedup_first_file[dedup_sig] = buffer;
    }
  } else if (dedup_skip) {
    /* Relaunch of an already-traced kernel: replay the first instance's
     * trace so the kernelslist remains a complete launch sequence. */
    fprintf(kernelsFile, "%s\n",
            ptrace_dedup_first_file[dedup_sig].c_str());
  }
  fclose(kernelsFile);

  statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
  unsigned blocks = gridDimX * gridDimY * gridDimZ;
  unsigned threads = blockDimX * blockDimY * blockDimZ;

  fprintf(statsFile, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", buffer,
          nvbit_get_func_name(ctx, func, true), gridDimX, gridDimY, gridDimZ,
          blocks, blockDimX, blockDimY, blockDimZ, threads);

  fclose(statsFile);

  ctx_kernelid[ctx]++;
  ctx_current_kernel_name[ctx] =
      std::string(nvbit_get_func_name(ctx, func, true));
  recv_thread_receiving = true;
}

static void leave_kernel_launch(CUcontext ctx, CUfunction func) {
  /* make sure current kernel is completed */
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda error: %s\n", cudaGetErrorName(err));
  }
  assert(err == cudaSuccess);

  /* make sure we prevent re-entry on the nvbit_callback when issuing
   * the flush_channel kernel */
  skip_flag = true;

  /* issue flush of channel so we are sure all the memory accesses
   * have been pushed */
  flush_channel<<<1, 1>>>();
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);

  /* unset the skip flag */
  skip_flag = false;

  /* wait here until the receiving thread has not finished with the
   * current kernel */
  while (recv_thread_receiving) {
    pthread_yield();
  }

  unsigned total_insts_per_kernel =
      tool_ctrl.total - old_total_insts;
  old_total_insts = tool_ctrl.total;

  unsigned reported_insts_per_kernel =
      tool_ctrl.reported - old_total_reported_insts;
  old_total_reported_insts = tool_ctrl.reported;

  statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
  fprintf(statsFile, "%d,%d", total_insts_per_kernel,
          reported_insts_per_kernel);
  fprintf(statsFile, "\n");
  fclose(statsFile);

  /* .zst is already written by recv_thread via process_kernel_to_zst().
   * No file to close here. */

  std::string func_name = std::string(nvbit_get_func_name(ctx, func, true));
  if (active_from_start && !should_trace_kernel(ctx_kernelid[ctx], func_name))
    active_region = false;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;

  if (first_call == true) {
    first_call = false;
    std::string traces_folder = user_folder + "/traces";
    if (mkdir(traces_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ==
        -1) {
      if (errno == EEXIST) {
        // alredy exists
      } else {
        // something else
        std::cout << "cannot create folder error:" << strerror(errno)
                  << std::endl;
        return;
      }
    }
    kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "w");
    statsFile = fopen(ctx_stats_location[ctx].c_str(), "w");
    fprintf(statsFile,
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts\n");
    fclose(statsFile);
  }

  switch (cbid) {
  // We start with recording kernel launch events
  // Identify all the possible CUDA launch events without stream
  // parameters, they will not get involved with cuda graph
  case API_CUDA_cuLaunch:
  case API_CUDA_cuLaunchGrid: {
    cuLaunch_params *p = (cuLaunch_params *)params;
    CUfunction func = p->f;
    if (!is_exit) {
      enter_kernel_launch(ctx, func, cbid, params, false, false);
    } else {
      leave_kernel_launch(ctx, func);
    }
  } break;
  // To support kernel launched by cuda graph (in addition to existing kernel
  // launche method), we need to do:
  //
  // 1. instrument kernels at cudaGraphAddKernelNode event. This is for cases
  // that kernels are manually added to a cuda graph.
  // 2. distinguish captured kernels when kernels are recorded to a graph
  // using stream capture. cudaStreamIsCapturing() tells us whether a stream
  // is capturiong.
  // 3. per-kernel instruction counters, since cuda graph can launch multiple
  // kernels at the same time.
  //
  // Three cases:
  //
  // 1. original kernel launch:
  //     1a. for any kernel launch without using a stream, we instrument it
  //     before it is launched, call cudaDeviceSynchronize after it is
  //     launched and read the instruction counter of the kernel.
  //     1b. for any kernel launch using a stream, but the stream is not
  //     capturing, we do the same thing as 1a.
  //
  //  2. cuda graph using stream capturing: if a kernel is launched in a
  //  stream and the stream is capturing. We instrument the kernel before it
  //  is launched and do nothing after it is launched, because the kernel is
  //  not running until cudaGraphLaunch. Instead, we issue a
  //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
  //  instruction counters, since a cloned graph might be launched afterwards.
  //
  //  3. cuda graph manual: we instrument the kernel added by
  //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch as 2.
  //
  // The above method should handle most of cuda graph launch cases.
  // kernel launches with stream parameter, they can be used for cuda graph
  case API_CUDA_cuLaunchKernel_ptsz:
  case API_CUDA_cuLaunchKernel:
  case API_CUDA_cuLaunchCooperativeKernel:
  case API_CUDA_cuLaunchCooperativeKernel_ptsz:
  case API_CUDA_cuLaunchKernelEx:
  case API_CUDA_cuLaunchKernelEx_ptsz:
  case API_CUDA_cuLaunchGridAsync: {
    CUfunction func;
    CUstream hStream;

    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      func = p->f;
      hStream = p->config->hStream;
    } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
               cbid == API_CUDA_cuLaunchKernel ||
               cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
               cbid == API_CUDA_cuLaunchCooperativeKernel) {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      func = p->f;
      hStream = p->hStream;
    } else {
      cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
      func = p->f;
      hStream = p->hStream;
    }

    cudaStreamCaptureStatus streamStatus;
    /* check if the stream is capturing, if yes, do not sync */
    CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
    if (!is_exit) {
      bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
      enter_kernel_launch(ctx, func, cbid, params, stream_capture);
    } else {
      if (streamStatus != cudaStreamCaptureStatusActive) {
        if (verbose >= 1) {
          printf("kernel %s not captured by cuda graph\n",
                 nvbit_get_func_name(ctx, func));
        }
        leave_kernel_launch(ctx, func);
      } else {
        if (verbose >= 1) {
          printf("kernel %s captured by cuda graph\n",
                 nvbit_get_func_name(ctx, func));
        }
      }
    }
  } break;
  case API_CUDA_cuGraphAddKernelNode: {
    cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
    CUfunction func = p->nodeParams->func;

    if (!is_exit) {
      // cuGraphAddKernelNode_params->nodeParams is the same as
      // cuLaunchKernel_params up to sharedMemBytes
      enter_kernel_launch(ctx, func, cbid, (void *)p->nodeParams, false, true);
    }
  } break;
  case API_CUDA_cuGraphLaunch: {
    // if we are exiting a cuda graph launch:
    // Wait until the graph is completed using
    // cudaStreamSynchronize()
    if (is_exit) {
      cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;

      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      assert(cudaGetLastError() == cudaSuccess);
      /* push a flush channel kernel */
      skip_flag = true;
      flush_channel<<<1, 1, 0, p->hStream>>>();
      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      skip_flag = false;
      assert(cudaGetLastError() == cudaSuccess);
    }

  } break;
  // Now we need to record cuda memcpy events
  case API_CUDA_cuMemcpyHtoD_v2: {
    if (!is_exit) {
      cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
      char buffer[1024];
      kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
      sprintf(buffer, "MemcpyHtoD,0x%016llx,%llu", p->dstDevice, p->ByteCount);
      fprintf(kernelsFile, buffer);
      fprintf(kernelsFile, "\n");
      fclose(kernelsFile);
    }
  } break;
  // For cuProfiler, we need to set the active region accordingly
  case API_CUDA_cuProfilerStart: {
    if (is_exit && !active_from_start) {
      active_region = true;
    }
  } break;
  case API_CUDA_cuProfilerStop: {
    if (is_exit && !active_from_start) {
      active_region = false;
    }
  } break;
  default:
    break;
  }
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {
  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1))
        stride = addrs[s + 1] - addrs[s];
      else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        if (stride != addrs[s] - addrs[s - 1]) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

void base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {
  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      deltas.push_back(addrs[s] - last_address);
      last_address = addrs[s];
    }
  }
}

void trim_string(std::string &str) {
  // Remove the leading and trailing spaces
  str.erase(0, str.find_first_not_of(' '));
  str.erase(str.find_last_not_of(' ') + 1);
}

typedef std::map<uint32_t, size_t> counter_t;
typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> warp_key_t;

counter_t create_counter(const std::vector<uint32_t> &indices) {
  counter_t counter;
  for (auto instr_idx : indices) {
    counter[instr_idx] = 0;
  }
  return counter;
}

std::pair<std::string, std::vector<uint32_t>>
parse_spinlock_instructions(const std::string &line) {
  std::vector<uint32_t> indices;
  // Each line is of the form: <kernel_id>, <kernel_name>: <indices>
  // Though kernel id is not used
  // -2 for the comma and the space
  size_t name_length = line.find(':') - line.find(',') - 2;
  std::string kernel_name = line.substr(line.find(',') + 2, name_length);
  std::string indices_str = line.substr(line.find(':') + 1);
  trim_string(indices_str);
  std::stringstream ss(indices_str);
  std::string instr_idx;
  while (std::getline(ss, instr_idx, ' ')) {
    indices.push_back(std::stoi(instr_idx));
  }
  return {kernel_name, indices};
}

/* Format an instruction trace into the rest_of_line string that
 * post-traces-processing would capture.  This is everything AFTER the
 * cta_id_x cta_id_y cta_id_z warpid_tb prefix. */
std::string format_instruction(const inst_trace_t *ma) {
  std::ostringstream oss;

  /* Optional core-id fields (included in rest_of_line when enabled). */
  if (print_core_id) {
    oss << ma->sm_id << " " << ma->warpid_sm << " ";
  }

  /* Optional source-line info. */
  if (lineinfo) {
    oss << ma->line_num << " ";
  }

  /* Virtual PC (4-hex-digit). */
  char hex_buf[64];
  snprintf(hex_buf, sizeof(hex_buf), "%04x", ma->vpc);
  oss << hex_buf << " ";

  /* Active mask ANDed with predicate mask (8-hex-digit). */
  snprintf(hex_buf, sizeof(hex_buf), "%08x",
           ma->active_mask & ma->predicate_mask);
  oss << hex_buf << " ";

  /* Destination register. */
  if (ma->GPRDst >= 0) {
    oss << "1 R" << ma->GPRDst << " ";
  } else {
    oss << "0 ";
  }

  /* Opcode. */
  oss << id_to_opcode_map[ma->opcode_id] << " ";

  /* Source register count and registers. */
  unsigned src_count = 0;
  for (int s = 0; s < MAX_SRC; s++)
    if (ma->GPRSrcs[s] >= 0)
      src_count++;
  oss << src_count << " ";

  for (int s = 0; s < MAX_SRC; s++)
    if (ma->GPRSrcs[s] >= 0)
      oss << "R" << ma->GPRSrcs[s] << " ";

  /* Memory / address information. */
  std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
  if (ma->is_mem) {
    std::istringstream iss(id_to_opcode_map[ma->opcode_id]);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, '.'))
      if (!token.empty())
        tokens.push_back(token);
    oss << get_datawidth_from_opcode(tokens) << " ";

    bool base_stride_success = false;
    uint64_t base_addr = 0;
    int stride = 0;
    std::vector<long long> deltas;

    if (enable_compress) {
      base_stride_success =
          base_stride_compress(ma->addrs, mask, base_addr, stride);
      if (!base_stride_success) {
        base_delta_compress(ma->addrs, mask, base_addr, deltas);
      }
    }

    if (base_stride_success && enable_compress) {
      snprintf(hex_buf, sizeof(hex_buf), "0x%llx",
               (unsigned long long)base_addr);
      oss << (unsigned)address_format::base_stride << " " << hex_buf << " "
          << stride << " ";
    } else if (!base_stride_success && enable_compress) {
      snprintf(hex_buf, sizeof(hex_buf), "0x%llx",
               (unsigned long long)base_addr);
      oss << (unsigned)address_format::base_delta << " " << hex_buf << " ";
      for (size_t s = 0; s < deltas.size(); s++) {
        oss << deltas[s] << " ";
      }
    } else {
      oss << (unsigned)address_format::list_all << " ";
      for (int s = 0; s < 32; s++) {
        if (mask.test(s)) {
          snprintf(hex_buf, sizeof(hex_buf), "0x%016lx",
                   (unsigned long)ma->addrs[s]);
          oss << hex_buf << " ";
        }
      }
    }
  } else {
    oss << "0 ";
  }

  /* Immediate value — match fprintf("%d", ...) on uint64_t. */
  oss << (int)ma->imm << " ";

  return oss.str();
}

/* Build a .traceg-format string in memory, compress per-block with zstd,
 * and write a seekable .traceg.zst file.
 *
 * File format:
 *   [4-byte JSON header length (LE uint32)]
 *   [JSON header: {version, dict_size, num_blocks, header_size, block_offsets, frame_sizes}]
 *   [traceg header bytes]
 *   [dictionary bytes (= first block's raw text)]
 *   [zstd frame 0 (no dict)] [zstd frame 1 (with dict)] ... [zstd frame N-1]
 */
void process_kernel_to_zst(CUcontext ctx) {
  KernelHeader &hdr = ctx_kernel_header[ctx];
  TBBufferMap &tb_buffers = ctx_tb_buffers[ctx];

  /* ── Build .traceg header string ── */
  std::ostringstream header_oss;
  header_oss << "-kernel name = " << hdr.kernel_name << "\n";
  header_oss << "-kernel id = " << hdr.kernel_id << "\n";
  header_oss << "-grid dim = (" << hdr.gridDimX << "," << hdr.gridDimY << ","
             << hdr.gridDimZ << ")\n";
  header_oss << "-block dim = (" << hdr.blockDimX << "," << hdr.blockDimY << ","
             << hdr.blockDimZ << ")\n";
  header_oss << "-shmem = " << hdr.shmem << "\n";
  header_oss << "-nregs = " << hdr.nregs << "\n";
  header_oss << "-binary version = " << hdr.binary_version << "\n";
  header_oss << "-cuda stream id = " << hdr.cuda_stream_id << "\n";
  char addr_buf[64];
  snprintf(addr_buf, sizeof(addr_buf), "0x%016lx",
           (unsigned long)hdr.shmem_base_addr);
  header_oss << "-shmem base_addr = " << addr_buf << "\n";
  snprintf(addr_buf, sizeof(addr_buf), "0x%016lx",
           (unsigned long)hdr.local_mem_base_addr);
  header_oss << "-local mem base_addr = " << addr_buf << "\n";
  header_oss << "-nvbit version = " << NVBIT_VERSION << "\n";
  header_oss << "-accelsim tracer version = " << TRACER_VERSION << "\n";
  header_oss << "-enable lineinfo = " << lineinfo << "\n";
  header_oss << "\n";
  header_oss << "#traces format = [line_num] PC mask dest_num [reg_dests] "
                "opcode src_num "
                "[reg_srcs] mem_width [adrrescompress?] [mem_addresses] "
                "immediate\n";
  header_oss << "\n";
  std::string traceg_header = header_oss.str();

  /* ── Build per-block text and compress with zstd ── */
  unsigned total_blocks = hdr.gridDimX * hdr.gridDimY * hdr.gridDimZ;
  unsigned warps_per_block =
      (hdr.blockDimX * hdr.blockDimY * hdr.blockDimZ + 31) / 32;

  std::vector<std::string> block_texts;
  for (unsigned i = 0; i < total_blocks; i++) {
    unsigned z = i / (hdr.gridDimY * hdr.gridDimX);
    unsigned rem = i % (hdr.gridDimY * hdr.gridDimX);
    unsigned y = rem / hdr.gridDimX;
    unsigned x = rem % hdr.gridDimX;

    auto key = std::make_tuple((int)x, (int)y, (int)z);
    auto it = tb_buffers.find(key);

    if (it == tb_buffers.end() || it->second.empty())
      continue;

    auto &warps = it->second;
    std::ostringstream block_oss;
    block_oss << "\n#BEGIN_TB\n";
    block_oss << "\nthread block = " << x << "," << y << "," << z << "\n";

    for (unsigned w = 0; w < warps_per_block; w++) {
      block_oss << "\nwarp = " << w << "\n";
      if (w < warps.size()) {
        block_oss << "insts = " << warps[w].size() << "\n";
        for (auto &inst : warps[w]) {
          block_oss << inst << "\n";
        }
      } else {
        block_oss << "insts = 0\n";
      }
    }
    block_oss << "\n#END_TB\n";
    block_texts.push_back(block_oss.str());
  }

  if (block_texts.empty()) {
    fprintf(stderr, "Warning: no blocks for kernel %d\n", hdr.kernel_id);
    tb_buffers.clear();
    ctx_ldgsts_flags[ctx].clear();
    return;
  }

  /* Dictionary = first block's raw text. */
  const std::string &dict = block_texts[0];

  /* Compress each block as an independent zstd frame. */
  std::vector<std::vector<char>> frames(block_texts.size());
  for (size_t i = 0; i < block_texts.size(); i++) {
    const std::string &src = block_texts[i];
    size_t bound = ZSTD_compressBound(src.size());
    frames[i].resize(bound);

    size_t compressed_size;
    if (i == 0) {
      /* First block: compress without dictionary. */
      compressed_size =
          ZSTD_compress(frames[i].data(), bound, src.data(), src.size(), 3);
    } else {
      /* Subsequent blocks: compress with first block as dictionary. */
      ZSTD_CCtx *cctx = ZSTD_createCCtx();
      compressed_size = ZSTD_compress_usingDict(
          cctx, frames[i].data(), bound, src.data(), src.size(),
          dict.data(), dict.size(), 3);
      ZSTD_freeCCtx(cctx);
    }

    if (ZSTD_isError(compressed_size)) {
      fprintf(stderr, "Error: zstd compression failed for kernel %d block %zu: %s\n",
              hdr.kernel_id, i, ZSTD_getErrorName(compressed_size));
      tb_buffers.clear();
      ctx_ldgsts_flags[ctx].clear();
      return;
    }
    frames[i].resize(compressed_size);
  }

  /* Build JSON header. */
  std::ostringstream json;
  json << "{\"version\":1,\"dict_size\":" << dict.size()
       << ",\"num_blocks\":" << block_texts.size()
       << ",\"header_size\":" << traceg_header.size()
       << ",\"block_offsets\":[";
  size_t offset = 0;
  for (size_t i = 0; i < frames.size(); i++) {
    if (i > 0) json << ",";
    json << offset;
    offset += frames[i].size();
  }
  json << "],\"frame_sizes\":[";
  for (size_t i = 0; i < frames.size(); i++) {
    if (i > 0) json << ",";
    json << frames[i].size();
  }
  json << "]}";
  std::string json_str = json.str();

  /* Write output file. */
  char filename[2048];
  std::string trace_dir = user_folder + "/traces/";
  sprintf(filename, "%skernel-%d-ctx_0x%lx.trace.zst", trace_dir.c_str(),
          hdr.kernel_id, (unsigned long)ctx);
  FILE *f = fopen(filename, "wb");
  if (f) {
    /* 4-byte JSON header length (little-endian). */
    uint32_t json_len = (uint32_t)json_str.size();
    fwrite(&json_len, sizeof(json_len), 1, f);
    /* JSON header. */
    fwrite(json_str.data(), 1, json_str.size(), f);
    /* Traceg header. */
    fwrite(traceg_header.data(), 1, traceg_header.size(), f);
    /* Dictionary. */
    fwrite(dict.data(), 1, dict.size(), f);
    /* Compressed frames. */
    for (auto &frame : frames) {
      fwrite(frame.data(), 1, frame.size(), f);
    }
    fclose(f);

    size_t total_size = 4 + json_str.size() + traceg_header.size() +
                        dict.size() + offset;
    if (ptrace_sample_mode != 0) {
      printf("Wrote .zst: %s (%zu bytes, sampled %zu/%u blocks)\n", filename,
             total_size, block_texts.size(), total_blocks);
    } else {
      printf("Wrote .zst: %s (%zu bytes)\n", filename, total_size);
    }
  }

  /* Clear buffers for this context. */
  tb_buffers.clear();
  ctx_ldgsts_flags[ctx].clear();
}

void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  // This counter map will keep track of the spinlock instruction
  // count in the current detected spinlock loop for each warp
  // The detection start if a spinlock instruction is encountered (start to
  // increment the counter) and end when a non-spinlock instruction is
  // encountered (clear the counter)
  std::map<warp_key_t, counter_t> warp_counter_map;

  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;
    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

        /* when we get this cta_id_x it means the kernel has completed
         */
        if (ma->cta_id_x == -1) {
          if (!tool_ctrl.stop) {
            process_kernel_to_zst(ctx);
          }
          recv_thread_receiving = false;
          if (enable_spinlock_fast_forward) {
            // Clear the counter map for all warps as we are starting a new
            // kernel
            warp_counter_map.clear();
          }
          break;
        }

        /* Spinlock fast forwarding */
        if (enable_spinlock_fast_forward) {
          // Check if this warp is in the warp_counter_map
          warp_key_t warp_key = std::make_tuple(ma->cta_id_x, ma->cta_id_y,
                                                ma->cta_id_z, ma->warpid_tb);
          if (warp_counter_map.find(warp_key) == warp_counter_map.end()) {
            // This warp is not in the warp_counter_map, so we create a counter
            // for this warp using the spinlock instruction indices for the
            // current kernel
            std::vector<uint32_t> &indices =
                *(spinlock_instr_map[ctx_current_kernel_name[ctx]]);
            warp_counter_map[warp_key] = create_counter(indices);
          }

          // Get the counter map for this warp
          auto &counter = warp_counter_map[warp_key];

          // Now check if we should start spinlock fast forwarding for this warp
          if (counter.find(ma->instr_idx) != counter.end()) {
            // We are still in a spinlock loop, so we increment the counter
            counter[ma->instr_idx]++;
            if (counter[ma->instr_idx] > spinlock_iter_to_keep) {
              // This spinlock instruction is executed more than the threshold
              // so we fast forward it in the output trace
              // Note we are only fast forwarding the innermost spinlock loop
              num_processed_bytes += sizeof(inst_trace_t);
              continue;
            }
          } else {
            // We are exiting the innermost spinlock loop, so we reset the
            // counter map for this warp
            for (auto &[instr_idx, count] : counter) {
              count = 0;
            }
          }
        }

        /* Buffer the instruction for .zsim output */
        {
          std::string inst_line = format_instruction(ma);
          std::string opcode_str = id_to_opcode_map[ma->opcode_id];

          auto tb_key = std::make_tuple(ma->cta_id_x, ma->cta_id_y,
                                        ma->cta_id_z);

          /* LDGSTS dedup: one real LDGSTS produces two trace entries
           * (global + shared mem refs).  Match post-processing: skip
           * the first, keep the second of each pair. */
          if (opcode_str.find("LDGSTS") != std::string::npos) {
            auto ldgsts_key = std::make_tuple(
                ma->cta_id_x, ma->cta_id_y, ma->cta_id_z, ma->warpid_tb);
            bool &seen_first = ctx_ldgsts_flags[ctx][ldgsts_key];
            if (seen_first) {
              /* Second of the pair — keep it. */
              auto &warps = ctx_tb_buffers[ctx][tb_key];
              if ((unsigned)ma->warpid_tb >= warps.size())
                warps.resize(ma->warpid_tb + 1);
              warps[ma->warpid_tb].push_back(inst_line);
            }
            seen_first = !seen_first;
          } else {
            auto &warps = ctx_tb_buffers[ctx][tb_key];
            if ((unsigned)ma->warpid_tb >= warps.size())
              warps.resize(ma->warpid_tb + 1);
            warps[ma->warpid_tb].push_back(inst_line);
          }
        }

        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }
  free(recv_buffer);

  return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
  ctx_current_kernel_name[ctx] = "";
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, ctx);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
  }
  if (ptrace_dedup_limit > 0 && ptrace_dedup_skipped > 0) {
    printf("PTRACE_DEDUP: skipped %llu relaunches across %zu kernel "
           "signatures\n",
           (unsigned long long)ptrace_dedup_skipped,
           ptrace_dedup_count.size());
  }
}

void nvbit_at_ctx_init(CUcontext ctx) {
  // Everytime we init a context, add the foldername and kernelid to the set
  char buffer[2048];
  sprintf(buffer, "kernelslist_ctx_0x%lx", ctx);
  std::string tmp_kernelslist = user_folder + "/traces/" + buffer;
  ctx_kernelslist[ctx] = tmp_kernelslist;
  sprintf(buffer, "stats_ctx_0x%lx", ctx);
  std::string tmp_stats = user_folder + "/traces/" + buffer;
  ctx_stats_location[ctx] = tmp_stats;
  ctx_kernelid[ctx] = 1;
}