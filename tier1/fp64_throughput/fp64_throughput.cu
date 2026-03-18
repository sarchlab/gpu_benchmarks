/**
 * fp64_throughput.cu
 *
 * Microbenchmark for peak double-precision (FP64) floating-point compute
 * throughput on NVIDIA GPUs.
 *
 * Strategy:
 *   Each CUDA thread executes a long chain of fused multiply-add (FMA)
 *   instructions using the intrinsic fma(a, b, c).  Because each FMA counts
 *   as 2 floating-point operations (1 multiply + 1 add), the achieved GFLOPS
 *   is:
 *
 *       GFLOPS = (num_threads × fmas_per_thread × 2) / elapsed_seconds / 1e9
 *
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure FP64 arithmetic throughput
 *   rather than memory bandwidth.
 *
 * Note on FP64 throughput:
 *   On consumer GPUs (e.g. GeForce RTX series), FP64 throughput is typically
 *   1/32 or 1/64 of FP32 throughput.  On datacenter/HPC GPUs (e.g. A100,
 *   H100, V100) FP64 throughput is 1/2 of FP32 throughput.  Results will
 *   reflect the hardware capability of the device under test.
 *
 * Usage:
 *   ./fp64_throughput [fmas_per_thread]
 *
 *   fmas_per_thread  – optional number of FMA iterations per thread
 *                      (default 4096).  Increase if the kernel finishes in
 *                      under ~5 ms on a fast GPU, or decrease on consumer
 *                      GPUs where FP64 is rate-limited.
 */

#include <cstdio>
#include <cstdlib>

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

/** Default number of FMA operations per thread per kernel invocation. */
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
// FP64 FMA kernel
// ---------------------------------------------------------------------------

/**
 * Each thread performs `fmas_per_thread` fused multiply-add operations on
 * register-resident double values.  A dependency chain (each FMA uses the
 * result of the previous one) prevents the compiler from eliminating the
 * work as dead code while still allowing the hardware to keep its FP64
 * pipelines fully busy.
 *
 * A small set of independent accumulators (4) is used so that the compiler
 * can interleave FMA instructions and hide the FP64 pipeline latency,
 * achieving full issue-rate throughput.
 *
 * @param out              Device pointer to a single double — the kernel writes
 *                         a final checksum here to prevent dead-code elimination.
 * @param fmas_per_thread  Number of FMA iterations per thread.
 */
__global__ void fp64_fma_kernel(double *out, int fmas_per_thread)
{
    /* Use four independent accumulators so that the compiler can issue four
       FMAs per cycle and hide the FP64 pipeline latency on modern hardware. */
    double a0 = 1.0 + static_cast<double>(threadIdx.x) * 0.001;
    double a1 = a0 + 0.1;
    double a2 = a0 + 0.2;
    double a3 = a0 + 0.3;

    /* Multiplier and addend — kept != 1/0 so the compiler cannot constant-fold
       the entire loop away. */
    const double mul = 1.0000000001;
    const double add = 0.0000000001;

    for (int i = 0; i < fmas_per_thread; i += 4) {
        a0 = fma(a0, mul, add);
        a1 = fma(a1, mul, add);
        a2 = fma(a2, mul, add);
        a3 = fma(a3, mul, add);
    }

    /* Write a reduction of all accumulators to global memory so the compiler
       cannot elide the computation.  Only thread 0 of block 0 actually writes,
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
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("Device          : %s (id=%d)\n", prop.name, device_id);
    printf("SMs             : %d\n", prop.multiProcessorCount);
    printf("Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK,
           NUM_BLOCKS * THREADS_PER_BLOCK);
    printf("FMAs / thread   : %d\n", fmas_per_thread);
    printf("Kernel repeats  : %d\n\n", NUM_ITERATIONS);

    /* Allocate a single double on the device to receive the checksum. */
    double *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

    /* Create CUDA timing events. */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Warm-up launch — not timed. */
    fp64_fma_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out, fmas_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed benchmark loop. */
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        fp64_fma_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_out, fmas_per_thread);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

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
    printf("FP64 Throughput : %.2f GFLOPS\n", gflops);

    /* Clean up. */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_out));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
