/**
 * warp_shuffle.cu
 *
 * warp_shuffle microbenchmark.
  *
 * Microbenchmark for warp/wave shuffle throughput measuring four variants:
 *   - __shfl_xor   (butterfly / XOR shuffle)
 *   - __shfl       (indexed shuffle)
 *   - __shfl_up    (shift up)
 *   - __shfl_down  (shift down)
 *
 * Strategy:
 *   Each thread performs N shuffle operations in a tight loop, passing an
 *   integer value between lanes.  The measured throughput is the effective
 *   data movement rate:
 *
 *       GB/s = (num_threads × ops_per_thread × sizeof(int))
 *              / elapsed_seconds / 1e9
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
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_OPS_PER_THREAD  = 4096;
static const int DEFAULT_THREADS_PER_BLK = 256;
static const int DEFAULT_NUM_BLOCKS      = 1024;

/** Warp size (64 on AMD GCN/CDNA, 32 on NVIDIA). */
static const int WARP_SIZE = 32;

static const int DEFAULT_NUM_WARMUP = 0;

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
 * Run a single shuffle benchmark: warmup + timed iterations.
 * Returns average kernel time and Gops/s via out parameters.
 */
static void run_benchmark(const char *kernel_name,
                          KernelFunc kernel, int *d_out,
                          int ops_per_thread, int num_blocks,
                          int threads_per_block, const char *shuffle_variant,
                          int num_warmup,
                          double *out_avg_ms, double *out_gops)
{
    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
        kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_out, ops_per_thread);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Create CUDA timing events. */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Timed kernel launch. */
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
        kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_out, ops_per_thread);
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    long long num_threads  = static_cast<long long>(num_blocks)
                             * threads_per_block;
    long long total_ops    = num_threads
                             * static_cast<long long>(ops_per_thread);
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double gops            = static_cast<double>(total_ops) / elapsed_s / 1e9;

    

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f Gops/s\n",
            kernel_name, (double)elapsed_ms, (double)elapsed_ms, gops);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"ops_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d,\"shuffle_variant\":\"%s\"}}\n",
           kernel_name, (double)elapsed_ms, ops_per_thread, num_blocks,
           threads_per_block, shuffle_variant);

    *out_avg_ms = (double)elapsed_ms;
    *out_gops   = gops;

    /* Clean up events. */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
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

    int num_warmup = DEFAULT_NUM_WARMUP;
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    /* Print device information to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
            num_blocks, threads_per_block,
            num_blocks * threads_per_block);
    fprintf(stderr, "Warp size       : %d\n", WARP_SIZE);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Shuffle variant : %s\n", shuffle_variant);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    /* Allocate a single int on the device to receive the checksum. */
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    double total_time_ms = 0.0;
    double max_gops = 0.0;

    bool run_xor  = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_xor") == 0);
    bool run_basic = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl") == 0);
    bool run_up   = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_up") == 0);
    bool run_down = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_down") == 0);

    if (run_xor) {
        double avg_ms, gops;
        run_benchmark("shfl_xor_kernel",
                      (KernelFunc)shfl_xor_kernel, d_out,
                      ops_per_thread, num_blocks, threads_per_block, shuffle_variant,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_basic) {
        double avg_ms, gops;
        run_benchmark("shfl_kernel",
                      (KernelFunc)shfl_kernel, d_out,
                      ops_per_thread, num_blocks, threads_per_block, shuffle_variant,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_up) {
        double avg_ms, gops;
        run_benchmark("shfl_up_kernel",
                      (KernelFunc)shfl_up_kernel, d_out,
                      ops_per_thread, num_blocks, threads_per_block, shuffle_variant,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_down) {
        double avg_ms, gops;
        run_benchmark("shfl_down_kernel",
                      (KernelFunc)shfl_down_kernel, d_out,
                      ops_per_thread, num_blocks, threads_per_block, shuffle_variant,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"giga_ops_per_sec\",\"value\":%.2f}]}\n",
           total_time_ms, max_gops);

    /* Clean up. */
    CUDA_CHECK(cudaFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
