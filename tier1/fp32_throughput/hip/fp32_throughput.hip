/**
 * fp32_throughput.hip
 *
 * HIP port of the fp32_throughput microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for single-precision (FP32) floating-point compute
 * throughput on AMD and NVIDIA GPUs.
 *
 * Strategy:
 *   Each HIP thread executes a long chain of fused multiply-add (FMA)
 *   instructions using the intrinsic fmaf(a, b, c).  Because each FMA counts
 *   as 2 floating-point operations (1 multiply + 1 add), the achieved GFLOPS
 *   is:
 *
 *       GFLOPS = (num_threads × fmas_per_thread × 2) / elapsed_seconds / 1e9
 *
 *   The FMA loop uses registers. Every thread then stores one checksum, so
 *   the timed kernel includes a final store per thread as well as arithmetic.
 *   The store and launch overhead can matter at small work counts.
 *
 * Usage:
 *   ./fp32_throughput [fmas_per_thread]
 *
 *   fmas_per_thread  – optional number of FMA iterations per thread
 *                      (default 4096).  Increase if the kernel finishes in
 *                      under ~5 ms on a fast GPU.
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_fmas_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before the timed run)
 *
 * Measurement: the kernel is launched and timed EXACTLY ONCE per process
 * (repetition = fresh re-executions via `gpubench run --reps N`). Output:
 * one {"type":"kernel",...} JSON line carrying the GPU event time and host
 * wall-clock time for that single launch, then a summary whose
 * total_time_ms equals it. Derived metrics (GFLOPS) are intentionally not
 * computed here — derive them downstream from time and the work-descriptor
 * params.
 */

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "../../../common_hip/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

/** Default number of FMA operations per thread per kernel invocation. */
static const int DEFAULT_FMAS_PER_THREAD = 4096;

/** Number of threads per block (multiple of 64, fills a typical CU well). */
static const int DEFAULT_THREADS_PER_BLOCK = 256;

/**
 * Number of thread blocks.
 * Default 1 block = single-SM/CU measurement: FMA pipes are per-SM
 * resources; multiplying by SM count is an extrapolation, not a measured
 * device-wide rate or proven upper bound. Raise
 * BENCH_PARAM_num_blocks (e.g. >= 8 × SM count) for a device-wide run.
 */
static const int DEFAULT_NUM_BLOCKS = 1;

// ---------------------------------------------------------------------------
// FP32 FMA kernel
// ---------------------------------------------------------------------------

/**
 * Each thread performs `fmas_per_thread` fused multiply-add operations on
 * register-resident float values. Each accumulator forms a dependency chain;
 * the per-thread output makes all four chains observable. Compiled-target
 * instruction counts must still be checked before deriving throughput.
 *
 * A small set of independent accumulators (UNROLL=4) is used so that the
 * compiler can interleave FMA instructions. The resulting issue rate depends
 * on the compiled instructions, occupancy, and target architecture.
 *
 * @param out            Device array with one float per launched thread.
 * @param fmas_per_thread  Number of FMA iterations per thread.
 */
__global__ void fp32_fma_kernel(float *out, int fmas_per_thread)
{
    /* Four independent accumulators expose instruction-level parallelism. */
    float a0 = 1.0f + static_cast<float>(threadIdx.x) * 0.001f;
    float a1 = a0 + 0.1f;
    float a2 = a0 + 0.2f;
    float a3 = a0 + 0.3f;

    /* Multiplier and addend — kept != 1/0 so the compiler cannot constant-fold
       the entire loop away. */
    const float mul = 1.0000001f;
    const float add = 0.0000001f;

    for (int i = 0; i < fmas_per_thread; i += 4) {
        a0 = fmaf(a0, mul, add);
        a1 = fmaf(a1, mul, add);
        a2 = fmaf(a2, mul, add);
        a3 = fmaf(a3, mul, add);
    }

    // One result per launched thread makes every accumulator chain observable.
    // This final store is included in the timed kernel, outside the FMA loop.
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    out[tid] = a0 + a1 + a2 + a3;
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
    int fmas_per_thread = DEFAULT_FMAS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_fmas_per_thread");
    if (env_val) {
        fmas_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        fmas_per_thread = atoi(argv[1]);
    }
    if (fmas_per_thread <= 0) {
        fprintf(stderr, "Invalid fmas_per_thread\n");
        return EXIT_FAILURE;
    }
    /* Round down to nearest multiple of 4 to match the loop unroll. */
    fmas_per_thread = (fmas_per_thread / 4) * 4;
    if (fmas_per_thread == 0) fmas_per_thread = 4;

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    // Validate before multiplying: the output owns one element per thread.
    if (num_blocks <= 0 || threads_per_block <= 0 ||
        static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(float) /
                static_cast<size_t>(threads_per_block)) {
        fprintf(stderr, "Invalid or overflowing launch dimensions\n");
        return EXIT_FAILURE;
    }
    const size_t total_threads =
        static_cast<size_t>(num_blocks) * static_cast<size_t>(threads_per_block);

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %zu total threads\n",
           num_blocks, threads_per_block,
           total_threads);
    fprintf(stderr, "FMAs / thread   : %d\n\n", fmas_per_thread);

    float *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(float)));

    char params[160];
    snprintf(params, sizeof(params),
             "\"fmas_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d",
             fmas_per_thread, num_blocks, threads_per_block);

    double total_gpu_ms =
        bench_run_once("fp32_fma_kernel", params, [&]() {
            fp32_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(
                d_out, fmas_per_thread);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
