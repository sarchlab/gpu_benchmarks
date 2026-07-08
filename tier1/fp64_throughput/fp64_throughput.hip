/**
 * fp64_throughput.hip
 *
 * HIP port of the fp64_throughput microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for peak double-precision (FP64) floating-point compute
 * throughput on AMD and NVIDIA GPUs.
 *
 * Strategy:
 *   Each HIP thread executes a long chain of fused multiply-add (FMA)
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
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_FMAS_PER_THREAD = 4096;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_BLOCKS = 1024;

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

    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

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
    fprintf(stderr, "FMAs / thread   : %d\n", fmas_per_thread);
    fprintf(stderr, "Warmup / iters  : %d / %d\n\n", num_warmup, num_iters);

    double *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(double)));

    /* Warm-up launches — not timed. */
    for (int w = 0; w < num_warmup; ++w) {
        fp64_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_out, fmas_per_thread);
        HIP_CHECK(hipDeviceSynchronize());
    }

    /* Timed iterations — each recorded individually. */
    bench_timer_t timer;
    bench_timer_create(&timer);

    char params[160];
    snprintf(params, sizeof(params),
             "\"fmas_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d",
             fmas_per_thread, num_blocks, threads_per_block);

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        bench_lap_begin(&timer, 0);
        fp64_fma_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_out, fmas_per_thread);
        bench_lap_t lap = bench_lap_end(&timer, 0);

        bench_emit_iter("fp64_fma_kernel", i, &lap, params);
        total_gpu_ms += lap.gpu_ms;
        fprintf(stderr, "iter %d: gpu = %.3f ms, host = %.3f ms\n",
                i, lap.gpu_ms, lap.host_ms);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    /* Clean up. */
    bench_timer_destroy(&timer);
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
