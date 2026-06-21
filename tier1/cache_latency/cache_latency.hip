/**
 * cache_latency.hip
 *
 * HIP port of the cache_latency microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for GPU cache/memory latency using pointer chasing.
 *
 * Methodology:
 *   A single GPU thread walks a linked-list–style index chain where each
 *   visited element holds the index of the next element to visit. Because
 *   each load depends on the previous result (true data dependency), the
 *   hardware cannot overlap the loads — this directly exposes round-trip
 *   memory latency rather than bandwidth.
 *
 *   Chasing is done at CACHELINE granularity: only the first word of each
 *   cacheline is a chase node, so consecutive accesses land on distinct
 *   lines (no intra-line spatial hits) while the touched lines still span
 *   the whole array (cache footprint == array_bytes). num_accesses is
 *   derived as reuse_k * array_bytes/cacheline_bytes — K FULL laps, no cap:
 *   ~1 cold miss + (K-1) resident hits per line at sizes that fit a cache
 *   level, degrading to all-miss latency once the set exceeds the caches
 *   (L2 -> MALL -> DRAM). The count grows with size (largest points take
 *   minutes on HW). No separate cache warm-up; the timed kernel self-warms,
 *   and a 1-access primer absorbs code-object load.
 *
 * Output:
 *   Latency in nanoseconds per memory access.
 *
 * Usage:
 *   ./cache_latency [array_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_array_bytes
 *   BENCH_PARAM_cacheline_bytes   (default 64)
 *   BENCH_PARAM_reuse_k           (default 16)
 *   BENCH_PARAM_num_accesses      (default: derived; overrides the formula)
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

static const size_t   DEFAULT_ARRAY_BYTES     = 2ULL * 1024 * 1024;  // 2 MB
static const uint32_t DEFAULT_CACHELINE_BYTES = 64u;       // gfx942/MI300X line; one chase node per line
static const uint32_t DEFAULT_REUSE_K         = 16u;       // full laps over the set: ~1 cold miss + (K-1) hits per line
static const uint32_t DEFAULT_RNG_SEED        = 42u;

// ---------------------------------------------------------------------------
// Pointer-chasing kernel
// ---------------------------------------------------------------------------

/**
 * pointer_chase_kernel — single-threaded pointer-chasing latency measurement.
 */
__global__ void pointer_chase_kernel(
    const uint32_t * __restrict__ arr,
    uint32_t  start_idx,
    uint64_t  num_accesses,
    uint32_t *result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t idx = start_idx;

    for (uint64_t i = 0; i < num_accesses; ++i) {
        idx = arr[idx];
    }

    *result = idx;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/**
 * build_chain — random Hamiltonian cycle over cacheline-strided nodes.
 *
 * Only the first word of each cacheline is used as a chase node (element
 * indices {0, stride, 2*stride, ...}, stride = cacheline_bytes/4). Consecutive
 * accesses therefore land on distinct cachelines, so every chase step is a
 * fresh line lookup at the targeted level. The touched lines still span the
 * whole array, so the cache footprint == array_bytes.
 *
 *   arr      : host index array (array_bytes/4 elements; only node slots written)
 *   M        : number of cacheline nodes (== number of touched lines)
 *   stride   : element stride between nodes (== cacheline_bytes / 4)
 *   start_idx: out — element index of the first node
 *   seed     : RNG seed
 */
static void build_chain(uint32_t       *arr,
                        uint32_t        M,
                        uint32_t        stride,
                        uint32_t       &start_idx,
                        uint32_t        seed)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> perm(M);
    std::iota(perm.begin(), perm.end(), 0u);          // node ordinals 0..M-1
    std::shuffle(perm.begin(), perm.end(), rng);

    for (uint32_t i = 0; i < M; ++i) {
        uint32_t node      = perm[i] * stride;        // element index of this node
        uint32_t next_node = perm[(i + 1u) % M] * stride;
        arr[node] = next_node;
    }

    start_idx = perm[0] * stride;
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

    uint32_t cacheline_bytes = DEFAULT_CACHELINE_BYTES;
    env_val = getenv("BENCH_PARAM_cacheline_bytes");
    if (env_val) cacheline_bytes = (uint32_t)atoi(env_val);

    uint32_t reuse_k = DEFAULT_REUSE_K;
    env_val = getenv("BENCH_PARAM_reuse_k");
    if (env_val) reuse_k = (uint32_t)atoi(env_val);

    uint32_t rng_seed = DEFAULT_RNG_SEED;
    env_val = getenv("BENCH_PARAM_rng_seed");
    if (env_val) rng_seed = (uint32_t)atoi(env_val);

    const size_t N = array_bytes / sizeof(uint32_t);

    /* Cacheline-strided pointer chasing: one node per cacheline. */
    const uint32_t stride = cacheline_bytes / (uint32_t)sizeof(uint32_t);
    if (stride == 0 || N < (size_t)stride * 2) {
        fprintf(stderr, "array_bytes (%zu) too small for cacheline_bytes (%u); "
                        "need >= 2 cachelines\n", array_bytes, cacheline_bytes);
        return EXIT_FAILURE;
    }
    const uint32_t M = (uint32_t)(N / stride);   // cacheline nodes == touched lines

    /* num_accesses = K FULL laps over the working set (no cap): ~1 cold miss +
     * (K-1) resident hits per line for sizes that fit a cache level; all-miss
     * latency once the set exceeds the caches. The count grows with array_bytes
     * (largest points take minutes on HW). An explicit override wins. */
    uint64_t num_accesses = (uint64_t)reuse_k * (uint64_t)M;
    if (num_accesses < 1u) num_accesses = 1u;
    env_val = getenv("BENCH_PARAM_num_accesses");
    if (env_val) num_accesses = strtoull(env_val, nullptr, 10);

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "=== cache_latency microbenchmark ===\n");
    fprintf(stderr, "Device         : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size     : %zu bytes (%zu KB)\n",
            array_bytes, array_bytes / 1024);
    fprintf(stderr, "Cacheline      : %u B  (stride %u elems)\n", cacheline_bytes, stride);
    fprintf(stderr, "Lines (nodes)  : %u\n", M);
    fprintf(stderr, "Reuse K / accs : %u laps -> %llu accesses\n",
            reuse_k, (unsigned long long)num_accesses);
    fprintf(stderr, "RNG seed       : %u\n", rng_seed);
    fprintf(stderr, "Method         : single-thread cacheline-stride pointer chasing\n\n");

    /* Build chain on host (only the M node slots are written; rest stay 0). */
    std::vector<uint32_t> h_arr(N, 0u);
    uint32_t start_idx = 0;
    build_chain(h_arr.data(), M, stride, start_idx, rng_seed);

    /* Allocate device memory. */
    uint32_t *d_arr    = nullptr;
    uint32_t *d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_arr,    array_bytes));
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint32_t)));

    /* Upload chain. */
    HIP_CHECK(hipMemcpy(d_arr, h_arr.data(), array_bytes,
                        hipMemcpyHostToDevice));

    /* Module-load primer: one untimed launch with a single access so the timed
     * launch below doesn't pay the one-time code-object load (each data point is
     * a fresh process). It touches only the start line, so the working set stays
     * cold — the timed kernel self-warms (first lap misses, K-1 laps hit). */
    hipLaunchKernelGGL(pointer_chase_kernel, dim3(1), dim3(1), 0, 0,
                       d_arr, start_idx, 1u, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    /* Timed runs. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start, 0));
    hipLaunchKernelGGL(pointer_chase_kernel, dim3(1), dim3(1), 0, 0,
                       d_arr, start_idx, num_accesses, d_result);
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));

    /* Latency (ns) = total_time_ns / num_accesses */
    const double latency_ns = static_cast<double>(total_ms) * 1e6
                              / static_cast<double>(num_accesses);

    fprintf(stderr, "Array %zu KB: latency = %.2f ns\n",
            array_bytes / 1024, latency_ns);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"pointer_chase_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"array_bytes\":%zu,\"num_accesses\":%llu,"
           "\"cacheline_bytes\":%u,\"reuse_k\":%u,\"rng_seed\":%u}}\n",
           (double)total_ms, array_bytes, (unsigned long long)num_accesses,
           cacheline_bytes, reuse_k, rng_seed);

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
