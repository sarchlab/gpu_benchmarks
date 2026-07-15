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
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before the timed run)
 *
 * Measurement: the kernel is launched and timed EXACTLY ONCE per process
 * (repetition = fresh re-executions via `gpubench run --reps N`). Output:
 * one {"type":"kernel",...} JSON line carrying the GPU event time and host
 * wall-clock time for that single launch, then a summary whose
 * total_time_ms equals it. Derived metrics (GIOPS) are intentionally not
 * computed here — derive them downstream from time and the work-descriptor
 * params.
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
// Default 1 block = single-SM/CU measurement: integer ALUs are per-SM
// resources, so device peak ≈ per-SM value × SM count (an upper bound —
// an all-SM load throttles below single-SM boost clock).  Raise
// BENCH_PARAM_num_blocks (e.g. >= 8 × SM count) for a device-wide run.
static const int DEFAULT_NUM_BLOCKS = 1;

// ---------------------------------------------------------------------------
// INT32 MAD kernel
// ---------------------------------------------------------------------------

// mul/add arrive as runtime kernel arguments (not compile-time constants) on
// purpose: a constant multiplier lets the optimizer strength-reduce the
// multiply to shifts/adds and even algebraically collapse the constant
// recurrence a <- a*mul+add into fewer instructions (nvcc folds four
// `a*3+1` steps into a single a*81+40 IMAD), which would report far more MADs
// than the hardware executes. The arithmetic is unsigned so overflow wraps
// (well-defined) rather than being signed-overflow UB.
__global__ void int32_mad_kernel(unsigned *out, int mads_per_thread,
                                 unsigned mul, unsigned add)
{
    unsigned a0 = 1u + threadIdx.x;
    unsigned a1 = a0 + 111u;
    unsigned a2 = a0 + 222u;
    unsigned a3 = a0 + 333u;

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

    unsigned *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned)));

    // Numerical Recipes LCG constants, supplied at runtime so the multiply is a
    // genuine hardware IMAD (see kernel comment).
    const unsigned mad_mul = 1664525u;
    const unsigned mad_add = 1013904223u;

    char params[160];
    snprintf(params, sizeof(params),
             "\"mads_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d",
             mads_per_thread, num_blocks, threads_per_block);

    double total_gpu_ms =
        bench_run_once("int32_mad_kernel", params, [&]() {
            int32_mad_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(
                d_out, mads_per_thread, mad_mul, mad_add);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
