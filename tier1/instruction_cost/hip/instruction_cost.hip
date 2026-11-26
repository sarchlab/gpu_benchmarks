/**
 * instruction_cost.hip
 *
 * Host program of the instruction_cost microbenchmark (AMD gfx942 only).
 *
 * The timed kernels are hand-written assembly (instruction_cost_kernels.S),
 * assembled into a code object that the Makefile embeds in this executable.
 * The loop body of a kernel is exactly `ops_per_iter` copies of one
 * instruction followed by three loop-control instructions, so the time of a
 * launch is a count of known instructions rather than of whatever a compiler
 * chose to emit. Sweeping `ops_per_iter` separates the cost of one instruction
 * (slope) from the cost of the loop control (intercept).
 *
 * Environment variables:
 *   BENCH_PARAM_iterations         loop iterations per thread (>= 1)
 *   BENCH_PARAM_instruction        instruction placed in the loop body
 *   BENCH_PARAM_dependency         chained | independent
 *   BENCH_PARAM_ops_per_iter       copies of the instruction per iteration;
 *                                  0 runs the loop control alone
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_vector_add_warmup  (default true; see bench_timing.h)
 *
 * Measurement: the selected kernel is launched and timed EXACTLY ONCE per
 * process. After the timed bracket every thread's stored result is compared
 * with a CPU reference; a mismatch is a fatal error. Output: one
 * {"type":"kernel",...} JSON line named after the launched kernel, then a
 * summary.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../../../common_hip/bench_timing.h"
#include "instruction_cost_kernels.h"

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

/* The assembled code object, embedded by the Makefile (ld -r -b binary). */
extern "C" const unsigned char _binary_instruction_cost_kernels_hsaco_start[];

// ---------------------------------------------------------------------------
// The kernel set (single-sourced with the assembly)
// ---------------------------------------------------------------------------

#define IC_ENUMERATOR(instruction) OP_##instruction,
#define IC_NAME(instruction) #instruction,
#define IC_OPS_PER_ITER_VALUE(instruction, dependency, k) , k

enum Instruction { IC_FOR_EACH_INSTRUCTION(IC_ENUMERATOR) INSTRUCTION_COUNT };

static const char *const INSTRUCTION_NAMES[INSTRUCTION_COUNT] = {
    IC_FOR_EACH_INSTRUCTION(IC_NAME)};

/* Loop-body sizes that have a kernel; 0 is ic_empty. */
static const int OPS_PER_ITER_VALUES[] = {
    0 IC_FOR_EACH_OPS_PER_ITER(IC_OPS_PER_ITER_VALUE, , )};

static const int DEFAULT_ITERATIONS = 65536;
static const int DEFAULT_OPS_PER_ITER = 4;
/** One wavefront: the per-instruction cost of a wavefront running alone. */
static const int DEFAULT_THREADS_PER_BLOCK = 64;
static const int DEFAULT_NUM_BLOCKS = 1;

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static float f32_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t bits_from_f32(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double f64_from_bits(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t bits_from_f64(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t pair(uint32_t lo, uint32_t hi)
{
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

/**
 * The 64-bit value a kernel stores per thread: X after `applications`
 * applications of X = op(X, ...) from the initial operands in
 * instruction_cost_kernels.h. GPU and CPU both round to nearest and fuse the
 * FMAs, so the comparison is bit-exact.
 */
static uint64_t reference_result(Instruction instruction, uint64_t applications)
{
    const float f32_m = f32_from_bits(IC_F32_M);
    const float f32_a = f32_from_bits(IC_F32_A);
    switch (instruction) {
    case OP_v_fma_f32: {
        float x = f32_from_bits(IC_F32_X0);
        for (uint64_t i = 0; i < applications; ++i) x = fmaf(x, f32_m, f32_a);
        return bits_from_f32(x);
    }
    case OP_v_mul_f32: {
        float x = f32_from_bits(IC_F32_X0);
        for (uint64_t i = 0; i < applications; ++i) x = x * f32_m;
        return bits_from_f32(x);
    }
    case OP_v_add_f32: {
        float x = f32_from_bits(IC_F32_X0);
        for (uint64_t i = 0; i < applications; ++i) x = x + f32_a;
        return bits_from_f32(x);
    }
    case OP_v_mul_lo_u32: {
        uint32_t x = IC_U32_X0;
        for (uint64_t i = 0; i < applications; ++i) x *= IC_U32_M;
        return x;
    }
    case OP_v_add_u32: {
        uint32_t x = IC_U32_X0;
        for (uint64_t i = 0; i < applications; ++i) x += IC_U32_A;
        return x;
    }
    case OP_v_fma_f64: {
        const double m = f64_from_bits(pair(IC_F64_M_LO, IC_F64_M_HI));
        const double a = f64_from_bits(pair(IC_F64_A_LO, IC_F64_A_HI));
        double x = f64_from_bits(pair(IC_F64_X0_LO, IC_F64_X0_HI));
        for (uint64_t i = 0; i < applications; ++i) x = fma(x, m, a);
        return bits_from_f64(x);
    }
    case OP_v_pk_fma_f32: {
        float lo = f32_from_bits(IC_F32_X0);
        float hi = f32_from_bits(IC_F32_X0_HI);
        for (uint64_t i = 0; i < applications; ++i) {
            lo = fmaf(lo, f32_m, f32_a);
            hi = fmaf(hi, f32_m, f32_a);
        }
        return pair(bits_from_f32(lo), bits_from_f32(hi));
    }
    case OP_v_mov_b64:
    case INSTRUCTION_COUNT:
        break;
    }
    return pair(IC_B64_LO, IC_B64_HI); /* a move leaves X unchanged */
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

static int int_param(const char *name, int fallback)
{
    const char *value = getenv(name);
    return value ? atoi(value) : fallback;
}

static const char *string_param(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return value ? value : fallback;
}

static void fail(const char *message)
{
    printf("{\"type\":\"error\",\"severity\":\"fatal\",\"message\":\"%s\"}\n", message);
    fprintf(stderr, "FAIL: %s\n", message);
    exit(EXIT_FAILURE);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const int iterations = int_param("BENCH_PARAM_iterations", DEFAULT_ITERATIONS);
    const char *instruction_name = string_param("BENCH_PARAM_instruction", INSTRUCTION_NAMES[0]);
    const char *dependency = string_param("BENCH_PARAM_dependency", "chained");
    const int ops_per_iter = int_param("BENCH_PARAM_ops_per_iter", DEFAULT_OPS_PER_ITER);
    const int num_blocks = int_param("BENCH_PARAM_num_blocks", DEFAULT_NUM_BLOCKS);
    const int threads_per_block =
        int_param("BENCH_PARAM_threads_per_block", DEFAULT_THREADS_PER_BLOCK);

    int instruction = 0;
    while (instruction < INSTRUCTION_COUNT &&
           strcmp(instruction_name, INSTRUCTION_NAMES[instruction]) != 0) {
        ++instruction;
    }
    if (instruction == INSTRUCTION_COUNT) fail("Unknown instruction");

    const bool chained = strcmp(dependency, "chained") == 0;
    if (!chained && strcmp(dependency, "independent") != 0) fail("Unknown dependency");

    bool has_kernel = false;
    for (int k : OPS_PER_ITER_VALUES) has_kernel = has_kernel || k == ops_per_iter;
    if (!has_kernel) fail("No kernel for this ops_per_iter");

    /* The loop tests at the bottom, so a kernel runs at least one iteration. */
    if (iterations < 1) fail("iterations must be at least 1");

    /* 1024 is the workgroup size the kernels declare in their metadata. */
    if (num_blocks <= 0 || threads_per_block <= 0 || threads_per_block > 1024 ||
        static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(uint64_t) /
                static_cast<size_t>(threads_per_block)) {
        fail("Invalid or overflowing launch dimensions");
    }
    const size_t total_threads =
        static_cast<size_t>(num_blocks) * static_cast<size_t>(threads_per_block);

    /* Zero copies is the same kernel for every instruction and dependency. */
    char kernel_name[64];
    if (ops_per_iter == 0) {
        snprintf(kernel_name, sizeof(kernel_name), "ic_empty");
    } else {
        snprintf(kernel_name, sizeof(kernel_name), "ic_%s_%s_k%d", instruction_name,
                 dependency, ops_per_iter);
    }

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Kernel          : %s\n", kernel_name);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %zu total threads\n",
            num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "Iterations      : %d\n\n", iterations);

    hipModule_t module;
    HIP_CHECK(hipModuleLoadData(&module, _binary_instruction_cost_kernels_hsaco_start));
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, kernel_name));

    /* A sentinel no kernel stores, so a thread that stored nothing is caught. */
    uint64_t *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_out, 0xff, total_threads * sizeof(uint64_t)));

    char params[256];
    snprintf(params, sizeof(params),
             "\"iterations\":%d,\"instruction\":\"%s\",\"dependency\":\"%s\","
             "\"ops_per_iter\":%d,\"num_blocks\":%d,\"threads_per_block\":%d,"
             "\"verification\":\"full_cpu_v1\"",
             iterations, instruction_name, dependency, ops_per_iter, num_blocks,
             threads_per_block);

    int kernel_iterations = iterations;
    void *kernel_args[] = {&d_out, &kernel_iterations};
    const double total_gpu_ms = bench_run_once(kernel_name, params, [&]() {
        HIP_CHECK(hipModuleLaunchKernel(kernel, num_blocks, 1, 1, threads_per_block, 1, 1,
                                        0, 0, kernel_args, nullptr));
    });

    /* Chained, X saw every instruction the loop executed. Independent, the
     * stored register holds one application whatever the iteration count.
     * ic_empty stores the untouched v_mov_b64 operand. */
    uint64_t expected = reference_result(OP_v_mov_b64, 0);
    if (ops_per_iter > 0) {
        const uint64_t applications =
            chained ? static_cast<uint64_t>(ops_per_iter) * static_cast<uint64_t>(iterations)
                    : 1;
        expected = reference_result(static_cast<Instruction>(instruction), applications);
    }

    std::vector<uint64_t> out(total_threads);
    HIP_CHECK(hipMemcpy(out.data(), d_out, total_threads * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < total_threads; ++i) {
        if (out[i] != expected) {
            fprintf(stderr, "thread %zu stored 0x%016llx, expected 0x%016llx\n", i,
                    static_cast<unsigned long long>(out[i]),
                    static_cast<unsigned long long>(expected));
            fail("Stored result differs from the CPU reference");
        }
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipModuleUnload(module));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
