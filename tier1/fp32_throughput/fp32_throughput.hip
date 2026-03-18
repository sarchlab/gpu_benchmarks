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

/** Default number of FMA operations per thread per kernel invocation. */
static const int DEFAULT_FMAS_PER_THREAD = 4096;

/** Number of threads per block (multiple of 64, fills a typical CU well). */
static const int THREADS_PER_BLOCK = 256;

/**
 * Number of thread blocks.
 * 1024 blocks × 256 threads = 262 144 concurrent threads — enough to
 * saturate even high-end GPUs with hundreds of CUs.
 */
static const int NUM_BLOCKS = 1024;

/** How many times to repeat the kernel for a stable average. */
static const int NUM_ITERATIONS = 10;

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
    /* Parse optional argument. */
    int fmas_per_thread = DEFAULT_FMAS_PER_THREAD;
    if (argc >= 2) {
        fmas_per_thread = atoi(argv[1]);
        if (fmas_per_thread <= 0) {
            fprintf(stderr, "Invalid fmas_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        /* Round down to nearest multiple of 4 to match the loop unroll. */
        fmas_per_thread = (fmas_per_thread / 4) * 4;
        if (fmas_per_thread == 0) fmas_per_thread = 4;
    }

    /* Print device information. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    printf("Device          : %s (id=%d)\n", prop.name, device_id);
    printf("CUs             : %d\n", prop.multiProcessorCount);
    printf("Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK,
           NUM_BLOCKS * THREADS_PER_BLOCK);
    printf("FMAs / thread   : %d\n", fmas_per_thread);
    printf("Kernel repeats  : %d\n\n", NUM_ITERATIONS);

    /* Allocate a single float on the device to receive the checksum. */
    float *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

    /* Create HIP timing events. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    /* Warm-up launch — not timed. */
    hipLaunchKernelGGL(fp32_fma_kernel, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                       0, 0, d_out, fmas_per_thread);
    HIP_CHECK(hipDeviceSynchronize());

    /* Timed benchmark loop. */
    HIP_CHECK(hipEventRecord(ev_start, 0));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        hipLaunchKernelGGL(fp32_fma_kernel, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                           0, 0, d_out, fmas_per_thread);
    }
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Compute GFLOPS.
     *   total_flops = iterations × num_threads × fmas_per_thread × 2
     *   (factor of 2: each FMA is 1 multiply + 1 add)
     */
    long long num_threads  = static_cast<long long>(NUM_BLOCKS)
                             * THREADS_PER_BLOCK;
    long long total_flops  = static_cast<long long>(NUM_ITERATIONS)
                             * num_threads
                             * static_cast<long long>(fmas_per_thread)
                             * 2LL;
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double gflops          = static_cast<double>(total_flops) / elapsed_s / 1e9;

    printf("Elapsed time    : %.3f ms (%.3f ms / kernel)\n",
           elapsed_ms, elapsed_ms / NUM_ITERATIONS);
    printf("Total FLOPs     : %lld\n", total_flops);
    printf("FP32 Throughput : %.2f GFLOPS\n", gflops);

    // CSV output (machine-parseable): benchmark_name,n_ops,time_ms,gflops
    printf("fp32_throughput,%lld,%.3f,%.2f\n", total_flops, (double)elapsed_ms, gflops);

    /* Clean up. */
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_out));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
