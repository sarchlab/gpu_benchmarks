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
 *   integer value between lanes.  The measured throughput is the effective
 *   data movement rate:
 *
 *       GB/s = (num_threads × ops_per_thread × sizeof(int))
 *              / elapsed_seconds / 1e9
 *
 *   A dependency chain (each shuffle uses the result of the previous one)
 *   prevents the compiler from eliminating the work as dead code while
 *   still reflecting the hardware's shuffle throughput.
 *
 * Usage:
 *   ./warp_shuffle [ops_per_thread]
 *
 *   ops_per_thread  – optional number of shuffle iterations per thread
 *                     (default 4096).
 */

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _err = (call);                                              \
        if (_err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Default number of shuffle operations per thread per kernel invocation. */
static const int DEFAULT_OPS_PER_THREAD = 4096;

/** Number of threads per block. */
static const int THREADS_PER_BLOCK = 256;

/** Number of thread blocks. */
static const int NUM_BLOCKS = 1024;

/** Warp size (64 on AMD GCN/CDNA, 32 on NVIDIA). */
#ifdef __HIP_PLATFORM_AMD__
static const int WARP_SIZE = 64;
#else
static const int WARP_SIZE = 32;
#endif

/** Number of warm-up iterations. */
static const int NUM_WARMUP = 5;

/** Number of timed iterations. */
static const int NUM_ITERATIONS = 5;

// ---------------------------------------------------------------------------
// Shuffle kernels
// ---------------------------------------------------------------------------

/**
 * shfl_xor_kernel — butterfly/XOR shuffle.
 * Each thread XOR-shuffles its value with lane (threadIdx.x ^ 1) in a
 * tight loop, creating a dependency chain.
 */
__global__ void shfl_xor_kernel(int *out, int ops_per_thread)
{
    int val = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = 0; i < ops_per_thread; ++i) {
        val = __shfl_xor(val, 1, WARP_SIZE);
    }
    /* Write result from thread 0 of block 0 to prevent dead-code elimination. */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = val;
    }
}

/**
 * shfl_kernel — indexed (direct) shuffle.
 * Each thread reads from lane (threadIdx.x ^ 1) using __shfl.
 */
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

/**
 * shfl_up_kernel — shift-up shuffle.
 * Each thread reads from the lane delta=1 below it.
 */
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

/**
 * shfl_down_kernel — shift-down shuffle.
 * Each thread reads from the lane delta=1 above it.
 */
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
 *
 * @param name          Human-readable name of the variant.
 * @param csv_variant   Short name for CSV output.
 * @param kernel        Function pointer to the __global__ kernel.
 * @param d_out         Device pointer to a single int (prevents DCE).
 * @param ops_per_thread Number of shuffle ops each thread executes.
 */
static void run_benchmark(const char *name, const char *csv_variant,
                          KernelFunc kernel, int *d_out,
                          int ops_per_thread)
{
    /* Warm-up launches. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        hipLaunchKernelGGL(kernel, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                           0, 0, d_out, ops_per_thread);
    }
    HIP_CHECK(hipDeviceSynchronize());

    /* Create HIP timing events. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    /* Timed benchmark loop. */
    HIP_CHECK(hipEventRecord(ev_start, 0));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        hipLaunchKernelGGL(kernel, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                           0, 0, d_out, ops_per_thread);
    }
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Compute throughput.
     *   data_moved = iterations * num_threads * ops_per_thread * sizeof(int)
     *   GB/s       = data_moved / elapsed_s / 1e9
     */
    long long num_threads  = static_cast<long long>(NUM_BLOCKS)
                             * THREADS_PER_BLOCK;
    long long total_ops    = static_cast<long long>(NUM_ITERATIONS)
                             * num_threads
                             * static_cast<long long>(ops_per_thread);
    double data_bytes      = static_cast<double>(total_ops) * sizeof(int);
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double gbs             = data_bytes / elapsed_s / 1e9;

    double avg_ms = elapsed_ms / NUM_ITERATIONS;

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f GB/s\n",
            name, (double)elapsed_ms, avg_ms, gbs);

    /* CSV output to stdout: warp_shuffle,<variant>,<time_ms>,<GBs> */
    printf("warp_shuffle,%s,%.3f,%.2f\n", csv_variant, avg_ms, gbs);

    /* Clean up events. */
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Parse optional argument. */
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
        if (ops_per_thread <= 0) {
            fprintf(stderr, "Invalid ops_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
            NUM_BLOCKS, THREADS_PER_BLOCK,
            NUM_BLOCKS * THREADS_PER_BLOCK);
    fprintf(stderr, "Warp size       : %d\n", WARP_SIZE);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n", NUM_WARMUP);
    fprintf(stderr, "Timed iters     : %d\n\n", NUM_ITERATIONS);

    /* Allocate a single int on the device to receive the checksum. */
    int *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    /* Run each shuffle variant benchmark. */
    run_benchmark("shfl_xor",  "shfl_xor",
                  (KernelFunc)shfl_xor_kernel,  d_out, ops_per_thread);
    run_benchmark("shfl",      "shfl",
                  (KernelFunc)shfl_kernel,       d_out, ops_per_thread);
    run_benchmark("shfl_up",   "shfl_up",
                  (KernelFunc)shfl_up_kernel,    d_out, ops_per_thread);
    run_benchmark("shfl_down", "shfl_down",
                  (KernelFunc)shfl_down_kernel,  d_out, ops_per_thread);

    /* Clean up. */
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
