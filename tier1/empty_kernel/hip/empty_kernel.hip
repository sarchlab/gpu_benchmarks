/**
 * empty_kernel.hip
 *
 * HIP port of the empty_kernel launch-overhead microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures the time to launch a single empty kernel. The kernel does no
 * work, so the measured time is pure launch / dispatch / grid-scheduling
 * cost. Sweeping num_blocks and block_size shows how that overhead grows
 * with grid geometry.
 *
 * Two timings per launch, recorded for every iteration:
 *   - host_time_ms : host wall-clock for launch + completion — the full
 *                    round trip as seen by the host.
 *   - time_ms      : GPU event time bracketing the launch — the
 *                    device-side dispatch / grid-scheduling cost.
 *
 * Usage:
 *   ./empty_kernel [num_blocks] [block_size]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_block_size
 *
 * Output: one {"type":"kernel",...} JSON line per timed iteration carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all iterations' GPU event times.
 */

#include <cstdio>
#include <cstdlib>

#include "../../../common_hip/bench_timing.h"

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

static const int DEFAULT_NUM_BLOCKS = 1024;
static const int DEFAULT_BLOCK_SIZE = 256;

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

    char params[96];
    snprintf(params, sizeof(params), "\"num_blocks\":%d,\"block_size\":%d",
             num_blocks, block_size);

    double total_gpu_ms =
        bench_run_once("empty_kernel", params, [&]() {
            empty_kernel<<<dim3(num_blocks), dim3(block_size)>>>();
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
