/**
 * int32_throughput.hip
 *
 * HIP port of the int32_throughput microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for peak 32-bit integer (INT32) compute throughput on
 * AMD and NVIDIA GPUs.
 *
 * Strategy:
 *   Each HIP thread executes a long chain of integer multiply-add (MAD)
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
 * Environment variables (override argv):
 *   BENCH_PARAM_mads_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 */

#include <cstdio>
#include <cstdlib>

#include "../../common/hip_cuda_compat.h"

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

static const int DEFAULT_MADS_PER_THREAD = 4096;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_BLOCKS = 1024;
static const int DEFAULT_NUM_WARMUP = 5;

// ---------------------------------------------------------------------------
// INT32 MAD kernel
// ---------------------------------------------------------------------------

__global__ void int32_mad_kernel(int *out, int mads_per_thread)
{
    int a0 = 1 + static_cast<int>(threadIdx.x);
    int a1 = a0 + 111;
    int a2 = a0 + 222;
    int a3 = a0 + 333;

    const int mul = 3;
    const int add = 1;

    for (int i = 0; i < mads_per_thread; i += 4) {
        a0 = a0 * mul + add;
        a1 = a1 * mul + add;
        a2 = a2 * mul + add;
        a3 = a3 * mul + add;
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

    int mads_per_thread = DEFAULT_MADS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_mads_per_thread");
    if (env_val) {
        mads_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        mads_per_thread = atoi(argv[1]);
    }
    if (mads_per_thread <= 0) {
        fprintf(stderr, "Invalid mads_per_thread\n");
        return EXIT_FAILURE;
    }
    mads_per_thread = (mads_per_thread / 4) * 4;
    if (mads_per_thread == 0) mads_per_thread = 4;

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    int num_warmup = DEFAULT_NUM_WARMUP;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block,
           num_blocks * threads_per_block);
    fprintf(stderr, "MADs / thread   : %d\n", mads_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    int *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
        int32_mad_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_out, mads_per_thread);
    }
    HIP_CHECK(hipDeviceSynchronize());

    /* Timed kernel launch. */
    HIP_CHECK(hipEventRecord(ev_start, 0));
    int32_mad_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_out, mads_per_thread);
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    long long num_threads  = static_cast<long long>(num_blocks) * threads_per_block;
    long long total_ops    = num_threads
                             * static_cast<long long>(mads_per_thread)
                             * 2LL;
    double elapsed_s       = static_cast<double>(elapsed_ms) * 1e-3;
    double giops           = static_cast<double>(total_ops) / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms\n", elapsed_ms);
    fprintf(stderr, "Total INT ops   : %lld\n", total_ops);
    fprintf(stderr, "INT32 Throughput : %.2f GIOPS\n", giops);

    /* JSON-lines output to stdout. */
    printf("{\"type\":\"kernel\",\"name\":\"int32_mad_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"mads_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d}}\n",
           (double)elapsed_ms, mads_per_thread, num_blocks, threads_per_block);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GIOPS\",\"value\":%.2f}]}\n",
           (double)elapsed_ms, giops);

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_out));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
