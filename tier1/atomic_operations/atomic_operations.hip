/**
 * atomic_operations.hip
 *
 * HIP port of the atomic_operations microbenchmark.
 * Compiles for both AMD (ROCm hipcc) and NVIDIA (nvcc -x cu) targets.
 *
 * Microbenchmark for atomic operation throughput on global memory:
 *   - atomicAdd
 *   - atomicCAS (compare-and-swap)
 *   - atomicMin
 *
 * Strategy:
 *   All threads perform N atomic operations on a single global memory
 *   location.  This creates maximum contention and measures the hardware
 *   atomic unit throughput.  The achieved Giga-Operations/s is:
 *
 *       Gops = (num_threads * ops_per_thread) / elapsed_seconds / 1e9
 *
 * Usage:
 *   ./atomic_operations [ops_per_thread]
 *
 *   ops_per_thread  – optional number of atomic ops per thread
 *                     (default 1024).
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_ops_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_atomic_op   ("atomicAdd", "atomicCAS", "atomicMin")
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__NVCC__) && !__has_include(<hip/hip_runtime.h>)
#include <cuda_runtime.h>
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemset               cudaMemset
#define hipMemsetAsync          cudaMemsetAsync
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyToSymbol       cudaMemcpyToSymbol
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetLastError         cudaGetLastError
#define hipGetErrorString       cudaGetErrorString
#define hipEvent_t              cudaEvent_t
#define hipEventCreate          cudaEventCreate
#define hipEventDestroy         cudaEventDestroy
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipStream_t             cudaStream_t
#define hipStreamCreate         cudaStreamCreate
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize
#define hipDeviceProp_t         cudaDeviceProp
#define hipError_t              cudaError_t
#define hipSuccess              cudaSuccess
#define hipHostMalloc           cudaMallocHost
#define hipHostFree             cudaFreeHost
#define hipHostMallocDefault    cudaHostAllocDefault
#define hipBlockIdx_x           blockIdx.x
#define hipBlockIdx_y           blockIdx.y
#define hipBlockIdx_z           blockIdx.z
#define hipThreadIdx_x          threadIdx.x
#define hipThreadIdx_y          threadIdx.y
#define hipThreadIdx_z          threadIdx.z
#define hipBlockDim_x           blockDim.x
#define hipBlockDim_y           blockDim.y
#define hipBlockDim_z           blockDim.z
#define hipGridDim_x            gridDim.x
#define hipGridDim_y            gridDim.y
#define hipGridDim_z            gridDim.z
#define __HIP_PLATFORM_AMD__    0
#else
#include <hip/hip_runtime.h>
#endif

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

static const int DEFAULT_OPS_PER_THREAD  = 1024;
static const int DEFAULT_THREADS_PER_BLK = 256;
static const int DEFAULT_NUM_BLOCKS      = 1024;
static const int DEFAULT_NUM_WARMUP      = 5;

// ---------------------------------------------------------------------------
// Atomic kernels
// ---------------------------------------------------------------------------

__global__ void atomicAdd_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicAdd(counter, 1u);
    }
}

__global__ void atomicCAS_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        unsigned int old = *counter;
        atomicCAS(counter, old, old + 1);
    }
}

__global__ void atomicMin_kernel(unsigned int *counter, int ops_per_thread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicMin(counter, tid + (unsigned int)i);
    }
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFunc)(unsigned int *, int);

/**
 * Run a single atomic benchmark: warmup + timed iterations.
 * Returns average kernel time in ms and Gops/s via out parameter.
 */
static void run_benchmark(const char *kernel_name,
                          KernelFunc kernel, unsigned int *d_counter,
                          int ops_per_thread, int num_blocks,
                          int threads_per_block, const char *atomic_op,
                          int num_warmup,
                          double *out_avg_ms, double *out_gops)
{
    /* Reset counter. */
    unsigned int zero = 0;
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(unsigned int),
                        hipMemcpyHostToDevice));

    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
        kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_counter, ops_per_thread);
    }
    HIP_CHECK(hipDeviceSynchronize());

    /* Reset counter before timed run. */
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(unsigned int),
                        hipMemcpyHostToDevice));

    /* Create HIP timing events. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    /* Timed kernel launch. */
    HIP_CHECK(hipEventRecord(ev_start, 0));
    kernel<<<dim3(num_blocks), dim3(threads_per_block)>>>(d_counter, ops_per_thread);
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    long long num_threads = static_cast<long long>(num_blocks)
                            * threads_per_block;
    long long total_ops   = num_threads
                            * static_cast<long long>(ops_per_thread);
    double elapsed_s      = static_cast<double>(elapsed_ms) * 1e-3;
    double gops           = static_cast<double>(total_ops) / elapsed_s / 1e9;

    

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f Gops/s\n",
            kernel_name, (double)elapsed_ms, (double)elapsed_ms, gops);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"ops_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d,\"atomic_op\":\"%s\"}}\n",
           kernel_name, (double)elapsed_ms, ops_per_thread, num_blocks,
           threads_per_block, atomic_op);

    *out_avg_ms = (double)elapsed_ms;
    *out_gops   = gops;

    /* Clean up events. */
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
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
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_ops_per_thread");
    if (env_val) {
        ops_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
    }
    if (ops_per_thread <= 0) {
        fprintf(stderr, "Invalid ops_per_thread\n");
        return EXIT_FAILURE;
    }

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    const char* atomic_op = "atomicAdd";
    env_val = getenv("BENCH_PARAM_atomic_op");
    if (env_val) atomic_op = env_val;
    if (strcmp(atomic_op, "atomicAdd") != 0 &&
        strcmp(atomic_op, "atomicCAS") != 0 &&
        strcmp(atomic_op, "atomicMin") != 0) {
        fprintf(stderr, "Error: unsupported atomic_op '%s'\n", atomic_op);
        return EXIT_FAILURE;
    }

    int num_warmup = DEFAULT_NUM_WARMUP;

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
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Atomic op       : %s\n", atomic_op);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    /* Allocate a single unsigned int on the device as the shared counter. */
    unsigned int *d_counter = nullptr;
    HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));

    double total_time_ms = 0.0;
    double max_gops = 0.0;

    bool run_add = (strcmp(atomic_op, "atomicAdd") == 0);
    bool run_cas = (strcmp(atomic_op, "atomicCAS") == 0);
    bool run_min = (strcmp(atomic_op, "atomicMin") == 0);

    if (run_add) {
        double avg_ms, gops;
        run_benchmark("atomicAdd_kernel",
                      (KernelFunc)atomicAdd_kernel, d_counter,
                      ops_per_thread, num_blocks, threads_per_block, atomic_op,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_cas) {
        double avg_ms, gops;
        run_benchmark("atomicCAS_kernel",
                      (KernelFunc)atomicCAS_kernel, d_counter,
                      ops_per_thread, num_blocks, threads_per_block, atomic_op,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_min) {
        double avg_ms, gops;
        run_benchmark("atomicMin_kernel",
                      (KernelFunc)atomicMin_kernel, d_counter,
                      ops_per_thread, num_blocks, threads_per_block, atomic_op,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"giga_ops_per_sec\",\"value\":%.2f}]}\n",
           total_time_ms, max_gops);

    /* Clean up. */
    HIP_CHECK(hipFree(d_counter));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
