/**
 * fp32_throughput.hip
 *
 * HIP port of the fp32_throughput microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for peak single-precision (FP32) floating-point compute
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
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure FP32 arithmetic throughput
 *   rather than memory bandwidth.
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
 *
 * Output: one {"type":"kernel",...} JSON line per timed iteration carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all iterations' GPU event times. Derived
 * metrics (GFLOPS) are intentionally not computed here — derive them
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

/** Default number of FMA operations per thread per kernel invocation. */
static const int DEFAULT_FMAS_PER_THREAD = 4096;

/** Number of threads per block (multiple of 64, fills a typical CU well). */
static const int DEFAULT_THREADS_PER_BLOCK = 256;

/**
 * Number of thread blocks.
 * 1024 blocks × 256 threads = 262 144 concurrent threads — enough to
 * saturate even high-end GPUs with hundreds of CUs.
 */
static const int DEFAULT_NUM_BLOCKS = 1024;

// ---------------------------------------------------------------------------
// FP32 FMA kernel
// ---------------------------------------------------------------------------

/**
 * Each thread performs `fmas_per_thread` fused multiply-add operations on
 * register-resident float values.  A dependency chain (each FMA uses the
 * result of the previous one) prevents the compiler from eliminating the
 * work as dead code while still allowing the hardware to keep its FP pipelines
 * fully busy.
 *
 * A small set of independent accumulators (UNROLL=4) is used so that the
 * compiler can interleave FMA instructions and hide the FP latency, fully
 * utilising the throughput path rather than the latency path.
 *
 * @param out            Device pointer to a single float — the kernel writes a
 *                       final checksum here to prevent dead-code elimination.
 * @param fmas_per_thread  Number of FMA iterations per thread.
 */
__global__ void fp32_fma_kernel(float *out, int fmas_per_thread)
{
    /* Use four independent accumulators so that the compiler can issue four
       FMAs per cycle and hide the 4-cycle FP32 latency on modern hardware. */
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

    /* Write a reduction of all accumulators to global memory so the compiler
       cannot elide the computation. Only thread 0 of block 0 actually writes,
       so the memory traffic is negligible. */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = a0 + a1 + a2 + a3;
    }
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
    fprintf(stderr, "FMAs / thread   : %d\n\n", fmas_per_thread);

    /* Allocate a single float on the device to receive the checksum. */
    float *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

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
