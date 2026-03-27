/**
 * fp16_throughput.cu
 *
 * CUDA native version of the fp16_throughput microbenchmark.
 * CUDA native implementation for NVIDIA GPUs.
 *
 * Measures peak half-precision (FP16) floating-point compute throughput in
 * GFLOPS (10^9 floating-point operations per second).
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
 *   NVIDIA GPU with FP16 support (sm_70 / Volta or later).
 *   Requires sm_70 (Volta) or later.
 *
 * Usage:
 *   ./fp16_throughput [fmas_per_thread]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_fmas_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 */

#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
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
static const int DEFAULT_NUM_WARMUP = 0;

// ---------------------------------------------------------------------------
// FP16 FMA kernel
// ---------------------------------------------------------------------------

__global__ void fp16_fma_kernel(__half2 *out, int fmas_per_thread)
{
    const float base = 1.0f + static_cast<float>(threadIdx.x) * 0.001f;
    __half2 a0 = __float2half2_rn(base);
    __half2 a1 = __float2half2_rn(base + 0.1f);
    __half2 a2 = __float2half2_rn(base + 0.2f);
    __half2 a3 = __float2half2_rn(base + 0.3f);

    const __half2 mul = __float2half2_rn(1.0000001f);
    const __half2 add = __float2half2_rn(0.0000001f);

    for (int i = 0; i < fmas_per_thread; i += 4) {
        a0 = __hfma2(a0, mul, add);
        a1 = __hfma2(a1, mul, add);
        a2 = __hfma2(a2, mul, add);
        a3 = __hfma2(a3, mul, add);
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = __hadd2(__hadd2(a0, a1), __hadd2(a2, a3));
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

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
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block,
           num_blocks * threads_per_block);
    fprintf(stderr, "FMAs / thread   : %d  (__hfma2, 4 FLOPs each)\n", fmas_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    __half2 *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(__half2)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int w = 0; w < num_warmup; ++w) {
        fp16_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_out, fmas_per_thread);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed kernel launch. */
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    fp16_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_out, fmas_per_thread);
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    long long num_threads = static_cast<long long>(num_blocks) * threads_per_block;
    long long total_flops = num_threads
                            * static_cast<long long>(fmas_per_thread)
                            * 4LL;
    double elapsed_s      = static_cast<double>(elapsed_ms) * 1e-3;
    double gflops         = static_cast<double>(total_flops) / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms\n", elapsed_ms);
    fprintf(stderr, "Total FLOPs     : %lld\n", total_flops);
    fprintf(stderr, "FP16 Throughput : %.2f GFLOPS\n", gflops);

    /* JSON-lines output to stdout. */
    printf("{\"type\":\"kernel\",\"name\":\"fp16_fma_kernel\",\"time_ms\":%.6f,"
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
