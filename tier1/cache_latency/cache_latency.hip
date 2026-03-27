/**
 * cache_latency.hip
 *
 * HIP port of the cache_latency microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for GPU cache/memory latency using pointer chasing.
 *
 * Methodology:
 *   A single GPU thread walks a linked-list–style index chain where
 *   each array element holds the index of the next element to visit.
 *   Because each load depends on the previous result (true data dependency),
 *   the hardware cannot overlap the loads — this directly exposes the
 *   round-trip memory latency rather than bandwidth.
 *
 * Output:
 *   Latency in nanoseconds per memory access.
 *
 * Usage:
 *   ./cache_latency [array_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_array_bytes
 *   BENCH_PARAM_num_accesses
 *   BENCH_PARAM_num_iters
 *   BENCH_PARAM_rng_seed
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

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
#define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#else
#include <hip/hip_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _err = (call);                                               \
        if (_err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const size_t   DEFAULT_ARRAY_BYTES  = 2ULL * 1024 * 1024;  // 2 MB
static const uint32_t DEFAULT_NUM_ACCESSES = 2000000u;
static const int      DEFAULT_NUM_ITERS    = 5;
static const uint32_t DEFAULT_RNG_SEED     = 42u;
static const int      DEFAULT_NUM_WARMUP   = 0;

// ---------------------------------------------------------------------------
// Pointer-chasing kernel
// ---------------------------------------------------------------------------

/**
 * pointer_chase_kernel — single-threaded pointer-chasing latency measurement.
 */
__global__ void pointer_chase_kernel(
    const uint32_t * __restrict__ arr,
    uint32_t  start_idx,
    uint32_t  num_accesses,
    uint32_t *result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t idx = start_idx;

    for (uint32_t i = 0; i < num_accesses; ++i) {
        idx = arr[idx];
    }

    *result = idx;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/**
 * build_chain — build a random Hamiltonian pointer-chasing cycle.
 */
static void build_chain(uint32_t       *arr,
                        uint32_t        N,
                        uint32_t       &start_idx,
                        uint32_t        seed)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (uint32_t i = 0; i < N; ++i) {
        arr[perm[i]] = perm[(i + 1u) % N];
    }

    start_idx = perm[0];
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
    long long array_bytes_ll = (long long)DEFAULT_ARRAY_BYTES;
    env_val = getenv("BENCH_PARAM_array_bytes");
    if (env_val) {
        array_bytes_ll = atoll(env_val);
    } else if (argc >= 2) {
        array_bytes_ll = atoll(argv[1]);
    }
    if (array_bytes_ll <= 0) {
        fprintf(stderr, "Invalid array_bytes\n");
        return EXIT_FAILURE;
    }
    size_t array_bytes = (size_t)array_bytes_ll;

    uint32_t num_accesses = DEFAULT_NUM_ACCESSES;
    env_val = getenv("BENCH_PARAM_num_accesses");
    if (env_val) num_accesses = (uint32_t)atoi(env_val);

    int num_iters = DEFAULT_NUM_ITERS;
    env_val = getenv("BENCH_PARAM_num_iters");
    if (env_val) num_iters = atoi(env_val);

    uint32_t rng_seed = DEFAULT_RNG_SEED;
    env_val = getenv("BENCH_PARAM_rng_seed");
    if (env_val) rng_seed = (uint32_t)atoi(env_val);

    int num_warmup = DEFAULT_NUM_WARMUP;
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    const size_t   N     = array_bytes / sizeof(uint32_t);
    const uint32_t N_u32 = static_cast<uint32_t>(N);

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "=== cache_latency microbenchmark ===\n");
    fprintf(stderr, "Device       : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size   : %zu bytes (%zu KB)\n",
            array_bytes, array_bytes / 1024);
    fprintf(stderr, "Num accesses : %u\n", num_accesses);
    fprintf(stderr, "Num iters    : %d\n", num_iters);
    fprintf(stderr, "RNG seed     : %u\n", rng_seed);
    fprintf(stderr, "Warmup iters : %d\n", num_warmup);
    fprintf(stderr, "Method       : single-thread pointer chasing\n\n");

    /* Build chain on host. */
    std::vector<uint32_t> h_arr(N);
    uint32_t start_idx = 0;
    build_chain(h_arr.data(), N_u32, start_idx, rng_seed);

    /* Allocate device memory. */
    uint32_t *d_arr    = nullptr;
    uint32_t *d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_arr,    array_bytes));
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint32_t)));

    /* Upload chain. */
    HIP_CHECK(hipMemcpy(d_arr, h_arr.data(), array_bytes,
                        hipMemcpyHostToDevice));

    /* Warm-up. */
    for (int w = 0; w < num_warmup; ++w) {
        hipLaunchKernelGGL(pointer_chase_kernel, dim3(1), dim3(1), 0, 0,
                           d_arr, start_idx, num_accesses, d_result);
    }
    HIP_CHECK(hipDeviceSynchronize());

    /* Timed runs. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start, 0));
    for (int iter = 0; iter < num_iters; ++iter) {
        hipLaunchKernelGGL(pointer_chase_kernel, dim3(1), dim3(1), 0, 0,
                           d_arr, start_idx, num_accesses, d_result);
    }
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));

    /* Latency (ns) = total_time_ns / total_accesses */
    const double total_accesses =
        static_cast<double>(num_accesses) * num_iters;
    const double latency_ns = static_cast<double>(total_ms) * 1e6 / total_accesses;
    const double avg_ms = (double)total_ms / num_iters;

    fprintf(stderr, "Array %zu KB: latency = %.2f ns\n",
            array_bytes / 1024, latency_ns);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"pointer_chase_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"array_bytes\":%zu,\"num_accesses\":%u,"
           "\"num_iters\":%d,\"rng_seed\":%u}}\n",
           avg_ms, array_bytes, num_accesses, num_iters, rng_seed);

    /* JSON-lines output to stdout: summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"latency_ns\",\"value\":%.2f}]}\n",
           (double)total_ms, latency_ns);

    /* Cleanup. */
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_arr));
    HIP_CHECK(hipFree(d_result));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
