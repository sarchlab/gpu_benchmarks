/**
 * atomic_operations.hip
 *
 * HIP port of the atomic_operations microbenchmark.
 * Compiles for both AMD (ROCm hipcc) and NVIDIA (nvcc -x cu) targets.
 *
 * Microbenchmark for atomic operation throughput on global memory:
 *   - atomicAdd
 *   - atomicCAS (compare-and-swap)
 *   - atomicMin
 *
 * Strategy:
 *   All threads perform N atomic operations on a single global memory
 *   location.  This creates maximum contention and measures the hardware
 *   atomic unit throughput.  The achieved Giga-Operations/s is:
 *
 *       Gops = (num_threads * ops_per_thread) / elapsed_seconds / 1e9
 *
 * Usage:
 *   ./atomic_operations [ops_per_thread]
 *
 *   ops_per_thread  – optional number of atomic ops per thread
 *                     (default 1024).
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_ops_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_atomic_op   ("atomicAdd", "atomicCAS", "atomicMin")
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before the timed run)
 *
 * Measurement: the selected kernel is launched and timed EXACTLY ONCE per
 * process (repetition = fresh re-executions via `gpubench run --reps N`).
 * Output: one {"type":"kernel",...} JSON line carrying the GPU event time
 * and host wall-clock time for that single launch, then a summary whose
 * total_time_ms equals it. Derived metrics (Gops/s) are intentionally not
 * computed here — derive them downstream from time and the work-descriptor
 * params.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../common/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_OPS_PER_THREAD  = 1024;
static const int DEFAULT_THREADS_PER_BLK = 256;
static const int DEFAULT_NUM_BLOCKS      = 2048;

// ---------------------------------------------------------------------------
// Atomic kernels
// ---------------------------------------------------------------------------

__global__ void atomicAdd_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicAdd(counter, 1u);
    }
}

__global__ void atomicCAS_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        unsigned int old = *counter;
        atomicCAS(counter, old, old + 1);
    }
}

__global__ void atomicMin_kernel(unsigned int *counter, int ops_per_thread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicMin(counter, tid + (unsigned int)i);
    }
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFunc)(unsigned int *, int);

/**
 * Run a single atomic benchmark: one timed launch (preceded by the vector-add
 * warmer unless disabled). Emits one kernel JSON line and returns that
 * launch's GPU event time in ms.
 */
static double run_benchmark(const char *kernel_name,
                            KernelFunc kernel, unsigned int *d_counter,
                            int ops_per_thread, int num_blocks,
                            int threads_per_block, const char *atomic_op)
{
    /* Reset counter once, outside the timed region. The counter's value does
     * not affect timing, so no further reset is needed. */
    unsigned int zero = 0;
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(unsigned int),
                        hipMemcpyHostToDevice));

    char params[192];
    snprintf(params, sizeof(params),
             "\"ops_per_thread\":%d,\"num_blocks\":%d,"
             "\"threads_per_block\":%d,\"atomic_op\":\"%s\"",
             ops_per_thread, num_blocks, threads_per_block, atomic_op);

    return bench_run_once(kernel_name, params, [&]() {
        kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_counter,
                                                              ops_per_thread);
    });
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters: env vars take priority, then argv, then defaults. */
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_ops_per_thread");
    if (env_val) {
        ops_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
    }
    if (ops_per_thread <= 0) {
        fprintf(stderr, "Invalid ops_per_thread\n");
        return EXIT_FAILURE;
    }

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    const char* atomic_op = "atomicAdd";
    env_val = getenv("BENCH_PARAM_atomic_op");
    if (env_val) atomic_op = env_val;
    if (strcmp(atomic_op, "atomicAdd") != 0 &&
        strcmp(atomic_op, "atomicCAS") != 0 &&
        strcmp(atomic_op, "atomicMin") != 0) {
        fprintf(stderr, "Error: unsupported atomic_op '%s'\n", atomic_op);
        return EXIT_FAILURE;
    }

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
            num_blocks, threads_per_block,
            num_blocks * threads_per_block);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Atomic op       : %s\n\n", atomic_op);

    /* Allocate a single unsigned int on the device as the shared counter. */
    unsigned int *d_counter = nullptr;
    HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));

    double total_gpu_ms = 0.0;

    bool run_add = (strcmp(atomic_op, "atomicAdd") == 0);
    bool run_cas = (strcmp(atomic_op, "atomicCAS") == 0);
    bool run_min = (strcmp(atomic_op, "atomicMin") == 0);

    if (run_add) {
        total_gpu_ms += run_benchmark("atomicAdd_kernel",
                                      (KernelFunc)atomicAdd_kernel, d_counter,
                                      ops_per_thread, num_blocks,
                                      threads_per_block, atomic_op);
    }

    if (run_cas) {
        total_gpu_ms += run_benchmark("atomicCAS_kernel",
                                      (KernelFunc)atomicCAS_kernel, d_counter,
                                      ops_per_thread, num_blocks,
                                      threads_per_block, atomic_op);
    }

    if (run_min) {
        total_gpu_ms += run_benchmark("atomicMin_kernel",
                                      (KernelFunc)atomicMin_kernel, d_counter,
                                      ops_per_thread, num_blocks,
                                      threads_per_block, atomic_op);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_counter));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
