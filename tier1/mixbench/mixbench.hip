/**
 * mixbench.hip
 *
 * Parametric roofline benchmark that varies compute-to-memory ratio across
 * multiple data points.  Inspired by github.com/ekondis/mixbench.
 *
 * Compiles for both AMD (ROCm/HIP) and NVIDIA (CUDA) targets via ifdefs.
 *
 * Strategy:
 *   Each GPU thread loads one float (4 bytes) from global memory, performs a
 *   variable-length chain of FP32 FMA operations, and stores the result back
 *   (4 bytes).  Total memory traffic per thread is 8 bytes (1 read + 1 write).
 *   By varying the number of FMA operations, we sweep the operational
 *   intensity (FLOP/byte) from memory-bound to compute-bound regimes,
 *   tracing out the roofline curve.
 *
 *   For a given flops_per_byte ratio r:
 *     num_fmas = r * 4   (since each FMA = 2 FLOP, total bytes = 8)
 *     achieved_gflops = (num_elements * num_fmas * 2) / time_s / 1e9
 *
 * Ratios tested: 0.5, 1, 2, 4, 8, 16, 32, 64
 *
 * Usage:
 *   ./mixbench [buffer_size_mb]
 *
 *   buffer_size_mb  – optional array size in MiB (default 256)
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_flops_per_byte
 *   BENCH_PARAM_threads_per_block
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Platform abstraction via ifdefs
// ---------------------------------------------------------------------------

#ifdef USE_CUDA

#include <cuda_runtime.h>

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define gpuMalloc              cudaMalloc
#define gpuFree                cudaFree
#define gpuMemcpy              cudaMemcpy
#define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuEventCreate         cudaEventCreate
#define gpuEventRecord         cudaEventRecord
#define gpuEventSynchronize    cudaEventSynchronize
#define gpuEventElapsedTime    cudaEventElapsedTime
#define gpuEventDestroy        cudaEventDestroy
#define gpuGetDevice           cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties

typedef cudaEvent_t      gpuEvent_t;
typedef cudaDeviceProp   gpuDeviceProp_t;

#else  /* USE_ROCM (default) */

#include <hip/hip_runtime.h>

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _err = (call);                                              \
        if (_err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define gpuMalloc              hipMalloc
#define gpuFree                hipFree
#define gpuMemcpy              hipMemcpy
#define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define gpuDeviceSynchronize   hipDeviceSynchronize
#define gpuEventCreate         hipEventCreate
#define gpuEventRecord         hipEventRecord
#define gpuEventSynchronize    hipEventSynchronize
#define gpuEventElapsedTime    hipEventElapsedTime
#define gpuEventDestroy        hipEventDestroy
#define gpuGetDevice           hipGetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties

typedef hipEvent_t       gpuEvent_t;
typedef hipDeviceProp_t  gpuDeviceProp_t;

#endif

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB       = 256;
static const double DEFAULT_FLOPS_PER_BYTE  = 4.0;
static const int    DEFAULT_THREADS_PER_BLK = 256;
static const int    DEFAULT_WARMUP_ITERS     = 0;

// ---------------------------------------------------------------------------
// Mixbench kernel
// ---------------------------------------------------------------------------

/**
 * Each thread loads one float from global memory, performs a variable-length
 * FMA chain, and stores the result.
 *
 * Memory per thread: 4 bytes read + 4 bytes write = 8 bytes total.
 * FLOPs per thread:  num_fmas * 2  (each FMA = 1 multiply + 1 add).
 * Operational intensity:  (num_fmas * 2) / 8  =  num_fmas / 4  FLOP/byte.
 *
 * @param data          Device array of floats (read/write)
 * @param num_elements  Number of elements in the array
 * @param num_fmas      Number of FMA operations per thread
 */
__global__ void mixbench_kernel(float *data, int num_elements, int num_fmas)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    float val = data[tid];

    /* Multiplier and addend chosen so the compiler cannot constant-fold the
       loop.  Values are close to 1 and 0 to avoid overflow/underflow. */
    const float mul = 1.0000001f;
    const float add = 0.0000001f;

    /* Dependency chain: each FMA uses the result of the previous one,
       preventing the compiler from eliminating the computation. */
    for (int i = 0; i < num_fmas; ++i) {
        val = fmaf(val, mul, add);
    }

    data[tid] = val;
}

// ---------------------------------------------------------------------------
// Benchmark one ratio
// ---------------------------------------------------------------------------

/**
 * Run the mixbench kernel at a given flops-per-byte ratio.
 *
 * @param d_data        Device data array
 * @param num_elements  Number of floats in the array
 * @param ratio         Target flops-per-byte ratio
 * @param threads_per_block Threads per block
 * @param[out] time_ms  Average kernel time in milliseconds
 * @param[out] gflops   Achieved GFLOPS
 * @param[out] bw_gbps  Achieved bandwidth in GB/s
 */
static void benchmark_ratio(float *d_data, int num_elements, double ratio,
                            int threads_per_block,
                            int warmup_iters,
                            double *time_ms, double *gflops, double *bw_gbps)
{
    /* Compute FMA count for this ratio.
       flops_per_byte = (num_fmas * 2) / 8  =>  num_fmas = ratio * 4 */
    int num_fmas = (int)(ratio * 4.0);
    if (num_fmas < 1) num_fmas = 1;

    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    /* Create GPU timing events. */
    gpuEvent_t ev_start, ev_stop;
    GPU_CHECK(gpuEventCreate(&ev_start));
    GPU_CHECK(gpuEventCreate(&ev_stop));

    /* Warmup iterations. */
    for (int i = 0; i < warmup_iters; ++i) {
        mixbench_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(
            d_data, num_elements, num_fmas);
    }
    GPU_CHECK(gpuDeviceSynchronize());

    /* Timed kernel launch. */
    GPU_CHECK(gpuEventRecord(ev_start, 0));
        mixbench_kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(
            d_data, num_elements, num_fmas);
    GPU_CHECK(gpuEventRecord(ev_stop, 0));
    GPU_CHECK(gpuEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    GPU_CHECK(gpuEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Average time per iteration. */
    double avg_ms = (double)elapsed_ms;

    /* Total FLOPs across all threads and all timed iterations:
       total_flops = num_elements * num_fmas * 2 */
    double total_flops = (double)num_elements
                         * (double)num_fmas
                         * 2.0;
    double elapsed_s = (double)elapsed_ms * 1e-3;
    double achieved_gflops = total_flops / elapsed_s / 1e9;

    /* Bandwidth: each element = 8 bytes (4 read + 4 write) */
    double total_bytes = (double)num_elements * 8.0;
    double achieved_bw = total_bytes / elapsed_s / 1e9;

    *time_ms = avg_ms;
    *gflops  = achieved_gflops;
    *bw_gbps = achieved_bw;

    GPU_CHECK(gpuEventDestroy(ev_start));
    GPU_CHECK(gpuEventDestroy(ev_stop));
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
    int buffer_mb = DEFAULT_BUFFER_MB;
    env_val = getenv("BENCH_PARAM_buffer_size_mb");
    if (env_val) {
        buffer_mb = atoi(env_val);
    } else if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
    }
    if (buffer_mb <= 0) {
        fprintf(stderr, "Invalid buffer_size_mb\n");
        return EXIT_FAILURE;
    }

    double flops_per_byte = DEFAULT_FLOPS_PER_BYTE;
    env_val = getenv("BENCH_PARAM_flops_per_byte");
    if (env_val) flops_per_byte = atof(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);


    int num_warmup = DEFAULT_WARMUP_ITERS;
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;
    int num_elements = (int)(bytes / sizeof(float));

    /* Print device information to stderr. */
    int device_id = 0;
    GPU_CHECK(gpuGetDevice(&device_id));
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size      : %d MiB (%d elements, %zu bytes)\n",
           buffer_mb, num_elements, bytes);
    fprintf(stderr, "Flops/byte      : %.1f\n", flops_per_byte);
    fprintf(stderr, "Threads/block   : %d\n", threads_per_block);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    /* Allocate and initialize host data. */
    float *h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory (%zu bytes)\n", bytes);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < num_elements; ++i) {
        h_data[i] = 1.0f + (float)(i % 1024) * 0.001f;
    }

    /* Allocate device memory and copy data. */
    float *d_data = nullptr;
    GPU_CHECK(gpuMalloc((void **)&d_data, bytes));
    GPU_CHECK(gpuMemcpy(d_data, h_data, bytes, gpuMemcpyHostToDevice));

    /* Run benchmark at the specified flops_per_byte ratio. */
    double avg_time_ms = 0.0;
    double achieved_gflops = 0.0;
    double achieved_bw = 0.0;

    /* Re-upload data to ensure consistent starting state. */
    GPU_CHECK(gpuMemcpy(d_data, h_data, bytes, gpuMemcpyHostToDevice));

    benchmark_ratio(d_data, num_elements, flops_per_byte,
                    threads_per_block,
                    num_warmup,
                    &avg_time_ms, &achieved_gflops, &achieved_bw);

    fprintf(stderr, "FLOP/byte=%.1f  Time=%.3f ms  GFLOPS=%.2f  BW=%.2f GB/s\n",
            flops_per_byte, avg_time_ms, achieved_gflops, achieved_bw);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"mixbench_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"buffer_size_mb\":%d,\"flops_per_byte\":%.1f,"
           "\"threads_per_block\":%d}}\n",
           avg_time_ms, buffer_mb, flops_per_byte, threads_per_block);

    /* JSON-lines output to stdout: summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f},"
           "{\"name\":\"bandwidth_gbps\",\"value\":%.2f}]}\n",
           avg_time_ms, achieved_gflops, achieved_bw);

    /* Clean up. */
    GPU_CHECK(gpuFree(d_data));
    free(h_data);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
