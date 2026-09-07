/**
 * warp_shuffle.hip
 *
 * HIP port of the warp_shuffle microbenchmark.
 * Compiles for both AMD (ROCm hipcc) and NVIDIA (nvcc -x cu) targets.
 *
 * Microbenchmark for warp/wave shuffle throughput measuring four variants:
 *   - __shfl_xor   (butterfly / XOR shuffle)
 *   - __shfl       (indexed shuffle)
 *   - __shfl_up    (shift up)
 *   - __shfl_down  (shift down)
 *
 * Strategy:
 *   Each thread performs N shuffle operations in a tight loop, passing an
 *   integer value between lanes.  The kernel is free of memory traffic, so
 *   the measured time reflects shuffle throughput.  The achieved rate in
 *   Giga-Operations/s is:
 *
 *       Gops = (num_threads * ops_per_thread) / elapsed_seconds / 1e9
 *
 *   (Optionally, an effective data-movement rate can be derived by
 *   multiplying by sizeof(int) = 4 bytes per shuffle.)
 *
 * Usage:
 *   ./warp_shuffle [ops_per_thread]
 *
 *   ops_per_thread  – optional number of shuffle iterations per thread
 *                     (default 4096).
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_ops_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_shuffle_variant   ("all", "shfl_xor", "shfl_up", "shfl_down", "shfl")
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before each timed run)
 *
 * Measurement: each selected variant's kernel is launched and timed EXACTLY
 * ONCE per process (repetition = fresh re-executions via
 * `gpubench run --reps N`). Output: one {"type":"kernel",...} JSON line per
 * variant that ran, carrying its GPU event time and host wall-clock time,
 * then a summary whose total_time_ms is the sum of those times. Derived
 * metrics (Gops/s) are intentionally not computed here — derive them
 * downstream from time and the work-descriptor params.
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

static const int DEFAULT_OPS_PER_THREAD  = 4096;
static const int DEFAULT_THREADS_PER_BLK = 256;
// Default 1 block = single-SM/CU, latency-bound measurement: each thread's
// shuffles form a serial dependent chain, so the result is the dependent-
// shuffle issue rate of one SM, not peak shuffle throughput.  Raising
// BENCH_PARAM_num_blocks adds warp-level parallelism but the per-thread
// chain remains dependent by design.
static const int DEFAULT_NUM_BLOCKS      = 1;

/** Warp size (64 on AMD GCN/CDNA, 32 on NVIDIA). */
#ifdef __HIP_PLATFORM_AMD__
static const int WARP_SIZE = 64;
#else
static const int WARP_SIZE = 32;
#endif

// ---------------------------------------------------------------------------
// Shuffle kernels
// ---------------------------------------------------------------------------

__global__ void shfl_xor_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_xor(val, 1, WARP_SIZE);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = val;
    }
}

__global__ void shfl_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    int src_lane = (threadIdx.x ^ 1) % WARP_SIZE;
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl(val, src_lane, WARP_SIZE);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = val;
    }
}

__global__ void shfl_up_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_up(val, 1, WARP_SIZE);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = val;
    }
}

__global__ void shfl_down_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_down(val, 1, WARP_SIZE);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = val;
    }
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFunc)(int *, int);

/**
 * Run a single shuffle variant: one timed launch (preceded by the vector-add
 * warmer unless disabled), emitted as its own kernel record. Returns that
 * launch's GPU event time in ms.
 */
static double run_benchmark(const char *kernel_name,
                            KernelFunc kernel, int *d_out,
                            int ops_per_thread, int num_blocks,
                            int threads_per_block, const char *shuffle_variant)
{
    char params[160];
    snprintf(params, sizeof(params),
             "\"ops_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d,"
             "\"shuffle_variant\":\"%s\"",
             ops_per_thread, num_blocks, threads_per_block, shuffle_variant);

    return bench_run_once(kernel_name, params, [&]() {
        kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_out,
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

    const char* shuffle_variant = "all";
    env_val = getenv("BENCH_PARAM_shuffle_variant");
    if (env_val) shuffle_variant = env_val;

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
    fprintf(stderr, "Warp size       : %d\n", WARP_SIZE);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Shuffle variant : %s\n\n", shuffle_variant);

    /* Allocate a single int on the device to receive the checksum. */
    int *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    bool run_xor  = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_xor") == 0);
    bool run_basic = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl") == 0);
    bool run_up   = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_up") == 0);
    bool run_down = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_down") == 0);

    double total_gpu_ms = 0.0;

    if (run_xor) {
        total_gpu_ms += run_benchmark("shfl_xor_kernel",
                                      (KernelFunc)shfl_xor_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant);
    }

    if (run_basic) {
        total_gpu_ms += run_benchmark("shfl_kernel",
                                      (KernelFunc)shfl_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant);
    }

    if (run_up) {
        total_gpu_ms += run_benchmark("shfl_up_kernel",
                                      (KernelFunc)shfl_up_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant);
    }

    if (run_down) {
        total_gpu_ms += run_benchmark("shfl_down_kernel",
                                      (KernelFunc)shfl_down_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
