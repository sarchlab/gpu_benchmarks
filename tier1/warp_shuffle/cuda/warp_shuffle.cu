/**
 * warp_shuffle.cu
 *
 * Native CUDA version of the warp_shuffle microbenchmark (rename-only port of ../hip/).
 * Compiles with nvcc for NVIDIA GPUs.
 *
 * Dependent warp/wave shuffle experiment measuring four variants:
 *   - __shfl_xor_sync   (butterfly / XOR shuffle)
 *   - __shfl_sync       (indexed shuffle)
 *   - __shfl_up_sync    (shift up)
 *   - __shfl_down_sync  (shift down)
 *
 * Strategy:
 *   Each thread performs N shuffle operations in a tight loop, passing an
 *   integer value between lanes. Every thread stores its final value. The
 *   event interval includes the dependent chain, loop/launch overhead and this
 *   one store; it is not an isolated instruction latency or peak throughput.
 *   Downstream logical shuffle rate = threads * ops_per_thread / GPU seconds.
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
#include <limits>

#include "../../../common_cuda/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_OPS_PER_THREAD  = 4096;
static const int DEFAULT_THREADS_PER_BLK = 256;
// A single block measures dependent chains with the selected warp-level
// parallelism. Multiplying its rate by SM/CU count does not establish a peak.
static const int DEFAULT_NUM_BLOCKS      = 1;

// ---------------------------------------------------------------------------
// Shuffle kernels
// ---------------------------------------------------------------------------

__global__ void shfl_xor_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_xor_sync(0xffffffff, val, 1, warpSize);
    }
    out[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = val;
}

__global__ void shfl_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    int src_lane = (threadIdx.x ^ 1) % warpSize;
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_sync(0xffffffff, val, src_lane, warpSize);
    }
    out[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = val;
}

__global__ void shfl_up_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_up_sync(0xffffffff, val, 1, warpSize);
    }
    out[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = val;
}

__global__ void shfl_down_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_down_sync(0xffffffff, val, 1, warpSize);
    }
    out[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = val;
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
                            int threads_per_block, const char *shuffle_variant,
                            int warp_size)
{
    char params[256];
    snprintf(params, sizeof(params),
             "\"ops_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d,"
             "\"shuffle_variant\":\"%s\",\"warp_size\":%d",
             ops_per_thread, num_blocks, threads_per_block, shuffle_variant, warp_size);

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

    const char* shuffle_variant = "shfl_xor";
    env_val = getenv("BENCH_PARAM_shuffle_variant");
    if (env_val) shuffle_variant = env_val;

    bool run_xor  = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_xor") == 0);
    bool run_basic = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl") == 0);
    bool run_up   = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_up") == 0);
    bool run_down = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_down") == 0);

    if (!run_xor && !run_basic && !run_up && !run_down) {
        fprintf(stderr, "Invalid shuffle_variant\n");
        return EXIT_FAILURE;
    }

    /* Print device information to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    // Full native waves keep every source lane active for all four intrinsics.
    // Bound both allocation arithmetic and the initial signed integer IDs.
    if ((prop.warpSize != 32 && prop.warpSize != 64) || num_blocks <= 0 ||
        num_blocks > prop.maxGridSize[0] || threads_per_block <= 0 ||
        threads_per_block > prop.maxThreadsPerBlock ||
        threads_per_block > prop.maxThreadsDim[0] ||
        threads_per_block % prop.warpSize != 0 ||
        static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(int) /
                static_cast<size_t>(threads_per_block) ||
        num_blocks > std::numeric_limits<int>::max() / threads_per_block) {
        fprintf(stderr, "Invalid launch geometry: require complete native waves "
                        "and representable per-thread output\n");
        return EXIT_FAILURE;
    }
    const size_t total_threads = static_cast<size_t>(num_blocks) * threads_per_block;

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %zu total threads\n",
            num_blocks, threads_per_block,
            total_threads);
    fprintf(stderr, "Warp size       : %d\n", prop.warpSize);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Shuffle variant : %s\n\n", shuffle_variant);

    /* One observable result for every launched thread. */
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total_threads * sizeof(int)));

    double total_gpu_ms = 0.0;

    if (run_xor) {
        total_gpu_ms += run_benchmark("shfl_xor_kernel",
                                      (KernelFunc)shfl_xor_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant, prop.warpSize);
    }

    if (run_basic) {
        total_gpu_ms += run_benchmark("shfl_kernel",
                                      (KernelFunc)shfl_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant, prop.warpSize);
    }

    if (run_up) {
        total_gpu_ms += run_benchmark("shfl_up_kernel",
                                      (KernelFunc)shfl_up_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant, prop.warpSize);
    }

    if (run_down) {
        total_gpu_ms += run_benchmark("shfl_down_kernel",
                                      (KernelFunc)shfl_down_kernel, d_out,
                                      ops_per_thread, num_blocks, threads_per_block,
                                      shuffle_variant, prop.warpSize);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    CUDA_CHECK(cudaFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
