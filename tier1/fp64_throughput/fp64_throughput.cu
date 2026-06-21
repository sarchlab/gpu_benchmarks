/**
 * fp64_throughput.cu
 *
 * CUDA native version of the fp64_throughput microbenchmark.
 * CUDA native implementation for NVIDIA GPUs.
 *
 * Microbenchmark for peak double-precision (FP64) floating-point compute
 * throughput on NVIDIA GPUs.
 *
 * Strategy:
 *   Each CUDA thread executes a long chain of fused multiply-add (FMA)
 *   instructions using the intrinsic fma(a, b, c) on double values.
 *   Because each FMA counts as 2 floating-point operations (1 multiply +
 *   1 add), the achieved GFLOPS is:
 *
 *       GFLOPS = (num_threads × fmas_per_thread × 2) / elapsed_seconds / 1e9
 *
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure FP64 arithmetic throughput
 *   rather than memory bandwidth.
 *
 * Note on FP64 throughput:
 *   On consumer GPUs, FP64 throughput is typically 1/32 or 1/64 of FP32
 *   throughput.  On datacenter/HPC GPUs (e.g. MI250X, A100, H100) FP64
 *   throughput approaches 1/2 of FP32 throughput.  Results will reflect the
 *   hardware capability of the device under test.
 *
 * Usage:
 *   ./fp64_throughput [fmas_per_thread]
 *
 *   fmas_per_thread  – optional number of FMA iterations per thread
 *                      (default 4096).  Increase if the kernel finishes in
 *                      under ~5 ms on a fast GPU, or decrease on consumer
 *                      GPUs where FP64 is rate-limited.
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_fmas_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_FMAS_PER_THREAD = 4096;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_BLOCKS = 1024;
static const int DEFAULT_NUM_WARMUP = 5;

// ---------------------------------------------------------------------------
// FP64 FMA kernel
// ---------------------------------------------------------------------------

__global__ void fp64_fma_kernel(double *out, int fmas_per_thread)
{
    double a0 = 1.0 + static_cast<double>(threadIdx.x) * 0.001;
    double a1 = a0 + 0.1;
    double a2 = a0 + 0.2;
    double a3 = a0 + 0.3;

    const double mul = 1.0000000001;
    const double add = 0.0000000001;

    for (int i = 0; i < fmas_per_thread; i += 4) {
        a0 = fma(a0, mul, add);
        a1 = fma(a1, mul, add);
        a2 = fma(a2, mul, add);
        a3 = fma(a3, mul, add);
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
    fmas_per_thread = (fmas_per_thread / 4) * 4;
    if (fmas_per_thread == 0) fmas_per_thread = 4;

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    int num_warmup = DEFAULT_NUM_WARMUP;

    /* Print device information to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block,
           num_blocks * threads_per_block);
    fprintf(stderr, "FMAs / thread   : %d\n", fmas_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    double *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
        fp64_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_out, fmas_per_thread);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed kernel launch. */
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
        fp64_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_out, fmas_per_thread);
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    long long num_threads  = static_cast<long long>(num_blocks) * threads_per_block;
    long long total_flops  = num_threads
                             * static_cast<long long>(fmas_per_thread)
                             * 2LL;
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double gflops          = static_cast<double>(total_flops) / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms\n", elapsed_ms);
    fprintf(stderr, "Total FLOPs     : %lld\n", total_flops);
    fprintf(stderr, "FP64 Throughput : %.2f GFLOPS\n", gflops);

    /* JSON-lines output to stdout. */
    printf("{\"type\":\"kernel\",\"name\":\"fp64_fma_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"fmas_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d}}\n",
           (double)elapsed_ms, fmas_per_thread, num_blocks, threads_per_block);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)elapsed_ms, gflops);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
