/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <stdint.h>

/* Tool control block: the three per-kernel counters/flags the injected
 * function needs, consolidated behind ONE pointer call-arg. NVBit 1.7.6
 * corrupts call args once the marshaled list exceeds ~26 32-bit register
 * units (the upstream tool sits exactly at 26); passing these three
 * individually plus the two sampling args overflows that and whichever arg
 * lands past the boundary arrives as garbage on the device. */
typedef struct {
  unsigned long long total;    // total dynamic warp instructions
  unsigned long long reported; // reported (recorded) warp instructions
  bool stop;                   // true = kernel not traced, count only
} tool_ctrl_t;

static __managed__ tool_ctrl_t tool_ctrl = {0, 0, false};

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
#define MAX_SRC 5

/* ── gpubench live block sampling (P1b) ──────────────────────────────────
 * The injected function early-exits for thread blocks outside the sample
 * plan, so unsampled blocks pay only a few comparisons instead of the full
 * address-gather + channel push. The mode word is baked into the injected
 * call as a constant at instrumentation time (it is process-wide, from the
 * PTRACE_SAMPLE env var); the explicit block list lives in managed memory.
 *
 * Mode bits:
 *   1 = corner sample: record blocks with ctaid.{x,y,z} < K (default K=2,
 *       set via "corner:K"; matches the Go-side cornerSample() in
 *       tracer/ptrace/sample.go). K lives in bits 8-15 of the mode word.
 *       Use K=3 for kernels with a boundary block at the origin so the
 *       fitter still sees two regular blocks per axis.
 *   2 = explicit list: record blocks listed in ptrace_sample_list.
 *   4 = count skipped blocks' warp instructions in the total counter
 *       (one atomicAdd per warp-inst; off by default for speed).
 */
#define PTRACE_SAMPLE_CORNER 1u
#define PTRACE_SAMPLE_LIST 2u
#define PTRACE_SAMPLE_COUNT_SKIPPED 4u
#define PTRACE_SAMPLE_CORNER_K(mode) (((mode) >> 8) & 0xffu)

#define PTRACE_MAX_SAMPLE_BLOCKS 64

/* The explicit list lives in plain cudaMalloc'd device memory, copied in at
 * the first kernel launch while the GPU is idle; the device pointer is baked
 * into the injected call as a constant. (A __managed__ static written by the
 * host faults with cudaErrorIllegalAddress when read from the injected
 * function under CUDA_MANAGED_FORCE_DEVICE_ALLOC — do not use UVM here.) */
typedef struct {
  int n;
  int x[PTRACE_MAX_SAMPLE_BLOCKS];
  int y[PTRACE_MAX_SAMPLE_BLOCKS];
  int z[PTRACE_MAX_SAMPLE_BLOCKS];
} ptrace_sample_list_t;

typedef struct {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warpid_tb;
  int warpid_sm;
  int sm_id;
  int opcode_id;
  uint64_t addrs[32];
  uint32_t line_num;
  uint32_t instr_idx;
  uint32_t vpc;
  bool is_mem;
  int32_t GPRDst;
  int32_t GPRSrcs[MAX_SRC];
  int32_t numSrcs;
  int32_t width;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint64_t imm;
} inst_trace_t;
