/**
 * int32_throughput.cu
 *
 * Microbenchmark for peak 32-bit integer (INT32) compute throughput on
 * NVIDIA GPUs.
 *
 * Strategy:
 *   Each CUDA thread executes a long chain of integer multiply-add (MAD)
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
 *   mads_per_thread  – optional number of MAD iterations per thread
 *                      (default 4096).  Increase if the kernel finishes in
 *                      under ~5 ms on a fast GPU.
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

/** Default number of MAD operations per thread per kernel invocation. */
static const int DEFAULT_MADS_PER_THREAD = 4096;

/** Number of threads per block (multiple of 32, fills a typical SM well). */
static const int THREADS_PER_BLOCK = 256;

/**
 * Number of thread blocks.
 * 1024 blocks × 256 threads = 262 144 concurrent threads — enough to
 * saturate even high-end GPUs with hundreds of SMs.
 */
static const int NUM_BLOCKS = 1024;

/** How many warmup iterations before timing. */
static const int NUM_WARMUP = 5;

/** How many times to repeat the kernel for a stable average. */
static const int NUM_ITERATIONS = 5;

// ---------------------------------------------------------------------------
// INT32 MAD kernel
// ---------------------------------------------------------------------------

/**
 * Each thread performs `mads_per_thread` integer multiply-add operations on
 * register-resident int values.  A dependency chain (each MAD uses the
 * result of the previous one) prevents the compiler from eliminating the
 * work as dead code while still allowing the hardware to keep its integer
 * pipelines fully busy.
 *
 * A small set of independent accumulators (UNROLL=4) is used so that the
 * compiler can interleave MAD instructions and hide the integer latency,
 * fully utilising the throughput path rather than the latency path.
 *
 * @param out              Device pointer to a single int — the kernel writes a
 *                         final checksum here to prevent dead-code elimination.
 * @param mads_per_thread  Number of MAD iterations per thread.
 */
__global__ void int32_mad_kernel(int *out, int mads_per_thread)
{
    /* Use four independent accumulators so that the compiler can issue four
       MADs per cycle and hide the integer latency on modern hardware. */
    int a0 = 1 + static_cast<int>(threadIdx.x);
    int a1 = a0 + 111;
    int a2 = a0 + 222;
    int a3 = a0 + 333;

    /* Multiplier and addend — kept != 1/0 so the compiler cannot constant-fold
       the entire loop away. */
    const int mul = 3;
    const int add = 1;

    for (int i = 0; i < mads_per_thread; i += 4) {
        a0 = a0 * mul + add;
        a1 = a1 * mul + add;
        a2 = a2 * mul + add;
        a3 = a3 * mul + add;
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
    int mads_per_thread = DEFAULT_MADS_PER_THREAD;
    if (argc >= 2) {
        mads_per_thread = atoi(argv[1]);
        if (mads_per_thread <= 0) {
            fprintf(stderr, "Invalid mads_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        /* Round down to nearest multiple of 4 to match the loop unroll. */
        mads_per_thread = (mads_per_thread / 4) * 4;
        if (mads_per_thread == 0) mads_per_thread = 4;
    }

    /* Print device information. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "SMs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK,
           NUM_BLOCKS * THREADS_PER_BLOCK);
    fprintf(stderr, "MADs / thread   : %d\n", mads_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n", NUM_WARMUP);
    fprintf(stderr, "Timed iters     : %d\n\n", NUM_ITERATIONS);

    /* Allocate a single int on the device to receive the checksum. */
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    /* Create CUDA timing events. */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Warm-up launches — not timed. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        int32_mad_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out, mads_per_thread);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed benchmark loop. */
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int32_mad_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_out, mads_per_thread);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Compute GIOPS.
     *   total_ops = iterations × num_threads × mads_per_thread × 2
     *   (factor of 2: each MAD is 1 multiply + 1 add)
     */
    long long num_threads  = static_cast<long long>(NUM_BLOCKS)
                             * THREADS_PER_BLOCK;
    long long total_ops    = static_cast<long long>(NUM_ITERATIONS)
                             * num_threads
                             * static_cast<long long>(mads_per_thread)
                             * 2LL;
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double giops           = static_cast<double>(total_ops) / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms (%.3f ms / kernel)\n",
           elapsed_ms, elapsed_ms / NUM_ITERATIONS);
    fprintf(stderr, "Total INT ops   : %lld\n", total_ops);
    fprintf(stderr, "INT32 Throughput : %.2f GIOPS\n", giops);

    // CSV output (machine-parseable): benchmark_name,n_ops,time_ms,giops
    printf("int32_throughput,%lld,%.3f,%.2f\n", total_ops, (double)elapsed_ms, giops);

    /* Clean up. */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
