/**
 * empty_kernel.hip
 *
 * HIP port of the empty_kernel launch-overhead microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for kernel launch overhead. An empty kernel does no work,
 * so the measured time is pure launch / dispatch / grid-scheduling cost.
 * Sweeping num_blocks (and block_size) shows how that overhead grows with
 * grid size.
 *
 * Two metrics (host wall-clock, since the cost is host-side; GPU events would
 * read ~0 for an empty kernel):
 *   - launch_sync_us : round-trip per launch (launch + device synchronize) —
 *                      full dispatch + schedule-empty-grid + completion latency.
 *   - enqueue_us     : back-to-back async dispatch per launch — host fire rate
 *                      (subject to queue backpressure once the grid is large).
 *
 * Usage:
 *   ./empty_kernel [num_blocks]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_block_size
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

#if defined(__NVCC__) && !__has_include(<hip/hip_runtime.h>)
#include <cuda_runtime.h>
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetLastError         cudaGetLastError
#define hipGetErrorString       cudaGetErrorString
#define hipError_t              cudaError_t
#define hipSuccess              cudaSuccess
#define hipDeviceProp_t         cudaDeviceProp
#define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#else
#include <hip/hip_runtime.h>
#endif

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _err = (call);                                               \
        if (_err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

using clock_type = std::chrono::high_resolution_clock;
static double us_since(const clock_type::time_point &t0)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock_type::now() - t0).count() / 1e3;
}

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int    DEFAULT_NUM_BLOCKS = 1024;
static const int    DEFAULT_BLOCK_SIZE = 256;
static const int    NUM_WARMUP         = 5;
static const double TARGET_US          = 50000.0;  // ~50 ms per measurement pass
static const int    MIN_ITERS          = 3;        // floor so ultra-slow giant grids stay quick
static const int    MAX_ITERS          = 50000;
static const int    CAL_ITERS          = 5;
static const int    MEASURE_PASSES     = 3;

// Pick an iteration count so a measurement pass takes ~TARGET_US, given a
// calibrated per-launch cost. Keeps every grid size measurable in ~constant
// wall-time (small grids run many iters; giant grids run few).
static int pick_iters(double per_launch_us)
{
    double v = TARGET_US / (per_launch_us > 0.01 ? per_launch_us : 0.01);
    if (v < MIN_ITERS) v = MIN_ITERS;
    if (v > MAX_ITERS) v = MAX_ITERS;
    return (int)v;
}

// ---------------------------------------------------------------------------
// Empty kernel
// ---------------------------------------------------------------------------

__global__ void empty_kernel() {}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *env_val;

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val)        num_blocks = atoi(env_val);
    else if (argc >= 2) num_blocks = atoi(argv[1]);
    if (num_blocks <= 0) { fprintf(stderr, "Invalid num_blocks\n"); return EXIT_FAILURE; }

    int block_size = DEFAULT_BLOCK_SIZE;
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    if (block_size <= 0 || block_size > 1024) { fprintf(stderr, "Invalid block_size\n"); return EXIT_FAILURE; }

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "=== empty_kernel launch-overhead microbenchmark ===\n");
    fprintf(stderr, "Device       : %s (id=%d, %d CUs)\n", prop.name, device_id, prop.multiProcessorCount);
    fprintf(stderr, "Grid         : %d blocks x %d threads\n", num_blocks, block_size);
    fprintf(stderr, "Warmup       : %d launches\n\n", NUM_WARMUP);

    /* Warm-up. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        hipLaunchKernelGGL(empty_kernel, dim3(num_blocks), dim3(block_size), 0, 0);
    }
    HIP_CHECK(hipDeviceSynchronize());

    /* Round-trip latency (launch + synchronize each iteration). */
    auto c0 = clock_type::now();
    for (int i = 0; i < CAL_ITERS; ++i) {
        hipLaunchKernelGGL(empty_kernel, dim3(num_blocks), dim3(block_size), 0, 0);
        HIP_CHECK(hipDeviceSynchronize());
    }
    const int rt_iters = pick_iters(us_since(c0) / CAL_ITERS);

    double launch_sync_us = -1.0;
    for (int pass = 0; pass < MEASURE_PASSES; ++pass) {
        auto t0 = clock_type::now();
        for (int i = 0; i < rt_iters; ++i) {
            hipLaunchKernelGGL(empty_kernel, dim3(num_blocks), dim3(block_size), 0, 0);
            HIP_CHECK(hipDeviceSynchronize());
        }
        double per = us_since(t0) / rt_iters;
        if (launch_sync_us < 0.0 || per < launch_sync_us) launch_sync_us = per;
    }

    /* Async enqueue cost (host dispatch only). Reuse rt_iters so the post-loop
       drain stays bounded even for giant grids, whose host-side enqueue is cheap
       but whose execution is not (enqueueing tens of thousands would then take
       minutes to drain). */
    double enqueue_us = -1.0;
    for (int pass = 0; pass < MEASURE_PASSES; ++pass) {
        auto t0 = clock_type::now();
        for (int i = 0; i < rt_iters; ++i) {
            hipLaunchKernelGGL(empty_kernel, dim3(num_blocks), dim3(block_size), 0, 0);
        }
        double per = us_since(t0) / rt_iters;
        HIP_CHECK(hipDeviceSynchronize());
        if (enqueue_us < 0.0 || per < enqueue_us) enqueue_us = per;
    }

    fprintf(stderr, "Iters        : %d per pass (adaptive)\n", rt_iters);

    const double total_ms = launch_sync_us * rt_iters / 1e3;

    fprintf(stderr, "Grid %d x %d: launch+sync = %.2f us, enqueue = %.2f us\n",
            num_blocks, block_size, launch_sync_us, enqueue_us);

    printf("{\"type\":\"kernel\",\"name\":\"empty_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"num_blocks\":%d,\"block_size\":%d}}\n",
           total_ms, num_blocks, block_size);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"launch_sync_us\",\"value\":%.3f},"
           "{\"name\":\"enqueue_us\",\"value\":%.3f}]}\n",
           total_ms, launch_sync_us, enqueue_us);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
