/**
 * fp16_throughput.cu
 *
 * Microbenchmark for peak half-precision (FP16) floating-point compute
 * throughput on NVIDIA GPUs (Volta / sm_70 and later).
 *
 * Strategy:
 *   Each CUDA thread executes a long chain of fused multiply-add (FMA)
 *   instructions using the __half2 packed type and the __hfma2() intrinsic.
 *   __half2 holds two fp16 values, so one __hfma2() call performs two FMAs
 *   simultaneously.  Counting 2 FLOPs per FMA:
 *
 *       FLOPs per __hfma2() = 2 (packed elements) × 2 (mul + add) = 4
 *
 *   GFLOPS = (num_threads × fmas_per_thread × 4) / elapsed_seconds / 1e9
 *
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure FP16 arithmetic throughput
 *   rather than memory bandwidth.
 *
 * Requirements:
 *   CUDA toolkit ≥ 9.0, GPU with compute capability ≥ 7.0 (Volta) for native
 *   FP16 tensor/CUDA-core throughput.
 *
 * Usage:
 *   ./fp16_throughput [fmas_per_thread]
 *
 *   fmas_per_thread  – optional number of __hfma2 iterations per thread
 *                      (default 4096).  Increase if the kernel finishes in
 *                      under ~5 ms on a fast GPU.
 */

#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Default number of __hfma2 operations per thread per kernel invocation. */
static const int DEFAULT_FMAS_PER_THREAD = 4096;

/** Number of threads per block (multiple of 32, fills a typical SM well). */
static const int THREADS_PER_BLOCK = 256;

/**
 * Number of thread blocks.
 * 1024 blocks × 256 threads = 262 144 concurrent threads — enough to
 * saturate even high-end GPUs with hundreds of SMs.
 */
static const int NUM_BLOCKS = 1024;

/** How many times to repeat the kernel for a stable average. */
static const int NUM_ITERATIONS = 10;

// ---------------------------------------------------------------------------
// FP16 FMA kernel
// ---------------------------------------------------------------------------

/**
 * Each thread performs `fmas_per_thread` packed half-precision fused
 * multiply-add operations on register-resident __half2 values.
 *
 * __hfma2(a, b, c) computes  { a.x*b.x + c.x,  a.y*b.y + c.y }  using the
 * hardware FP16 FMA unit.  Because both lanes execute simultaneously, each
 * call delivers 4 FLOPs.
 *
 * A dependency chain (each __hfma2 uses the previous result) prevents the
 * compiler from eliminating the work as dead code.  Four independent
 * accumulators (UNROLL=4) allow the compiler to interleave instructions and
 * hide the FP16 pipeline latency, achieving full issue-rate throughput.
 *
 * @param out            Device pointer to a single float — the kernel writes
 *                       a final checksum here to prevent dead-code elimination.
 * @param fmas_per_thread  Number of __hfma2 iterations per thread.
 */
__global__ void fp16_fma_kernel(__half2 *out, int fmas_per_thread)
{
    /* Initialise four independent __half2 accumulators from the thread index
       so the compiler cannot constant-fold the loop. */
    const float base = 1.0f + static_cast<float>(threadIdx.x) * 0.001f;
    __half2 a0 = __float2half2_rn(base);
    __half2 a1 = __float2half2_rn(base + 0.1f);
    __half2 a2 = __float2half2_rn(base + 0.2f);
    __half2 a3 = __float2half2_rn(base + 0.3f);

    /* Multiplier and addend — values close to 1/0 prevent saturation while
       staying far enough from zero that the compiler cannot fold the loop. */
    const __half2 mul = __float2half2_rn(1.0000001f);
    const __half2 add = __float2half2_rn(0.0000001f);

    /* Main compute loop.  Unrolled by 4 so the scheduler sees four independent
       FP16 FMA instructions per cycle, hiding the ~6-cycle FP16 latency. */
    for (int i = 0; i < fmas_per_thread; i += 4) {
        a0 = __hfma2(a0, mul, add);
        a1 = __hfma2(a1, mul, add);
        a2 = __hfma2(a2, mul, add);
        a3 = __hfma2(a3, mul, add);
    }

    /* Write a reduction of all accumulators to global memory so the compiler
       cannot elide the computation.  Only thread 0 of block 0 writes, so the
       memory traffic is negligible. */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = __hadd2(__hadd2(a0, a1), __hadd2(a2, a3));
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
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("Device          : %s (id=%d)\n", prop.name, device_id);
    printf("Compute cap.    : %d.%d\n", prop.major, prop.minor);
    printf("SMs             : %d\n", prop.multiProcessorCount);
    printf("Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK,
           NUM_BLOCKS * THREADS_PER_BLOCK);
    printf("FMAs / thread   : %d  (__hfma2, 4 FLOPs each)\n", fmas_per_thread);
    printf("Kernel repeats  : %d\n\n", NUM_ITERATIONS);

    /* Require sm_70 or later for native FP16 throughput. */
    if (prop.major < 7) {
        fprintf(stderr,
                "Warning: compute capability %d.%d detected. "
                "Native FP16 throughput requires sm_70 (Volta) or later.\n"
                "Results may not reflect hardware FP16 peak.\n",
                prop.major, prop.minor);
    }

    /* Allocate a single __half2 on the device to receive the checksum. */
    __half2 *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(__half2)));

    /* Create CUDA timing events. */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Warm-up launch — not timed. */
    fp16_fma_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out, fmas_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed benchmark loop. */
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        fp16_fma_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_out, fmas_per_thread);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Compute GFLOPS.
     *   Each __hfma2 call performs 2 packed FP16 FMAs.
     *   Each FMA = 2 FLOPs (multiply + add).
     *   => 4 FLOPs per __hfma2 call.
     *
     *   total_flops = iterations × num_threads × fmas_per_thread × 4
     */
    long long num_threads  = static_cast<long long>(NUM_BLOCKS)
                             * THREADS_PER_BLOCK;
    long long total_flops  = static_cast<long long>(NUM_ITERATIONS)
                             * num_threads
                             * static_cast<long long>(fmas_per_thread)
                             * 4LL;
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double gflops          = static_cast<double>(total_flops) / elapsed_s / 1e9;

    printf("Elapsed time    : %.3f ms (%.3f ms / kernel)\n",
           elapsed_ms, elapsed_ms / NUM_ITERATIONS);
    printf("Total FLOPs     : %lld\n", total_flops);
    printf("FP16 Throughput : %.2f GFLOPS\n", gflops);

    /* Clean up. */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_out));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
