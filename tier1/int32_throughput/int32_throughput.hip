/**
 * int32_throughput.hip
 *
 * HIP port of the int32_throughput microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for peak 32-bit integer (INT32) compute throughput on
 * AMD and NVIDIA GPUs.
 *
 * Strategy:
 *   Each HIP thread executes a long chain of integer multiply-add (MAD)
 *   instructions: a = a * mul + add.  Because each MAD counts as
 *   2 integer operations (1 multiply + 1 add), the achieved GIOPS is:
 *
 *       GIOPS = (num_threads × mads_per_thread × 2) / elapsed_seconds / 1e9
 *
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure INT32 arithmetic throughput
 *   rather than memory bandwidth.
 *
 * Usage:
 *   ./int32_throughput [mads_per_thread]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_mads_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *
 * Output: one {"type":"kernel",...} JSON line per timed iteration carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all iterations' GPU event times. Derived
 * metrics (GIOPS) are intentionally not computed here — derive them
 * downstream from time and the work-descriptor params.
 */

#include <cstdio>
#include <cstdlib>

#include "../../common/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_MADS_PER_THREAD = 4096;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_BLOCKS = 1024;

// ---------------------------------------------------------------------------
// INT32 MAD kernel
// ---------------------------------------------------------------------------

__global__ void int32_mad_kernel(int *out, int mads_per_thread)
{
    int a0 = 1 + static_cast<int>(threadIdx.x);
    int a1 = a0 + 111;
    int a2 = a0 + 222;
    int a3 = a0 + 333;

    const int mul = 3;
    const int add = 1;

    for (int i = 0; i < mads_per_thread; i += 4) {
        a0 = a0 * mul + add;
        a1 = a1 * mul + add;
        a2 = a2 * mul + add;
        a3 = a3 * mul + add;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = a0 + a1 + a2 + a3;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    int mads_per_thread = DEFAULT_MADS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_mads_per_thread");
    if (env_val) {
        mads_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        mads_per_thread = atoi(argv[1]);
    }
    if (mads_per_thread <= 0) {
        fprintf(stderr, "Invalid mads_per_thread\n");
        return EXIT_FAILURE;
    }
    mads_per_thread = (mads_per_thread / 4) * 4;
    if (mads_per_thread == 0) mads_per_thread = 4;

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block,
           num_blocks * threads_per_block);
    fprintf(stderr, "MADs / thread   : %d\n\n", mads_per_thread);

    int *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    char params[160];
    snprintf(params, sizeof(params),
             "\"mads_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d",
             mads_per_thread, num_blocks, threads_per_block);

    double total_gpu_ms =
        bench_run_once("int32_mad_kernel", params, [&]() {
            int32_mad_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(
                d_out, mads_per_thread);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
