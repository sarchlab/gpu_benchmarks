/**
 * empty_kernel.hip
 *
 * HIP port of the empty_kernel launch-overhead microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures the time to launch a single empty kernel, averaged over a fixed
 * number of iterations. The kernel does no work, so the measured time is
 * pure launch / dispatch / grid-scheduling cost. Sweeping num_blocks and
 * block_size shows how that overhead grows with grid geometry.
 *
 * Two timings per launch:
 *   - launch_sync_us : host wall-clock for launch + device synchronize —
 *                      the full round trip as seen by the host.
 *   - gpu_event_us   : GPU event time bracketing the launch — the
 *                      device-side dispatch / grid-scheduling cost.
 *
 * Usage:
 *   ./empty_kernel [num_blocks] [block_size]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_block_size
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>

#include "../../common/hip_cuda_compat.h"

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

static const int DEFAULT_NUM_BLOCKS = 1024;
static const int DEFAULT_BLOCK_SIZE = 256;
static const int NUM_WARMUP         = 10;
static const int NUM_ITERS          = 100;

__global__ void empty_kernel() {}

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
    if (env_val)        block_size = atoi(env_val);
    else if (argc >= 3) block_size = atoi(argv[2]);
    if (block_size <= 0 || block_size > 1024) { fprintf(stderr, "Invalid block_size\n"); return EXIT_FAILURE; }

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "=== empty_kernel launch-overhead microbenchmark ===\n");
    fprintf(stderr, "Device : %s (id=%d, %d CUs)\n", prop.name, device_id, prop.multiProcessorCount);
    fprintf(stderr, "Grid   : %d blocks x %d threads\n", num_blocks, block_size);
    fprintf(stderr, "Iters  : %d (+%d warmup)\n", NUM_ITERS, NUM_WARMUP);

    for (int w = 0; w < NUM_WARMUP; ++w) {
        empty_kernel<<<dim3(num_blocks), dim3(block_size)>>>();
        HIP_CHECK(hipDeviceSynchronize());
    }

    /* Host wall-clock: launch + synchronize round trip. */
    auto t0 = clock_type::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
        empty_kernel<<<dim3(num_blocks), dim3(block_size)>>>();
        HIP_CHECK(hipDeviceSynchronize());
    }
    const double launch_sync_us = us_since(t0) / NUM_ITERS;

    /* GPU events: device-side time bracketing each launch. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    double gpu_us_sum = 0.0;
    for (int i = 0; i < NUM_ITERS; ++i) {
        HIP_CHECK(hipEventRecord(ev_start, 0));
        empty_kernel<<<dim3(num_blocks), dim3(block_size)>>>();
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        gpu_us_sum += ms * 1e3;
    }
    const double gpu_event_us = gpu_us_sum / NUM_ITERS;

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    fprintf(stderr, "Grid %d x %d: launch+sync = %.2f us, gpu event = %.2f us\n",
            num_blocks, block_size, launch_sync_us, gpu_event_us);

    printf("{\"type\":\"kernel\",\"name\":\"empty_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"num_blocks\":%d,\"block_size\":%d}}\n",
           gpu_event_us / 1e3, num_blocks, block_size);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"launch_sync_us\",\"value\":%.3f},"
           "{\"name\":\"gpu_event_us\",\"value\":%.3f}]}\n",
           gpu_event_us / 1e3, launch_sync_us, gpu_event_us);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
