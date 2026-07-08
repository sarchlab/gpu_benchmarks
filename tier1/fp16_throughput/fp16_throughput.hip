/**
 * fp16_throughput.hip
 *
 * HIP port of the fp16_throughput microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures peak half-precision (FP16) floating-point compute throughput in
 * GFLOPS (10^9 floating-point operations per second).
 *
 * Strategy:
 *   Each HIP thread executes a long chain of fused multiply-add (FMA)
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
 *   ROCm/HIP with FP16 support (GCN5 / Vega and later, or any RDNA GPU).
 *   For NVIDIA targets via HIP-on-CUDA: sm_70 (Volta) or later.
 *
 * Usage:
 *   ./fp16_throughput [fmas_per_thread]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_fmas_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_num_warmup       (warmup launches, default 3)
 *   BENCH_PARAM_num_iters        (timed launches, each recorded, default 7)
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

static const int DEFAULT_FMAS_PER_THREAD = 4096;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_BLOCKS = 1024;

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

    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block,
           num_blocks * threads_per_block);
    fprintf(stderr, "FMAs / thread   : %d  (__hfma2, 4 FLOPs each)\n", fmas_per_thread);
    fprintf(stderr, "Warmup / iters  : %d / %d\n\n", num_warmup, num_iters);

    __half2 *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(__half2)));

    char params[160];
    snprintf(params, sizeof(params),
             "\"fmas_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d",
             fmas_per_thread, num_blocks, threads_per_block);

    /* Warmup + timed iterations, each recorded individually. */
    double total_gpu_ms =
        bench_run_loop("fp16_fma_kernel", num_warmup, num_iters, params, [&]() {
            fp16_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(
                d_out, fmas_per_thread);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    /* Clean up. */
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
