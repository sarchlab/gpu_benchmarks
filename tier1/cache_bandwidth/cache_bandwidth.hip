/**
 * cache_bandwidth.hip
 *
 * HIP port of the cache_bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for per-cache-level read bandwidth.
 *
 * Methodology:
 *   Each thread block streams (vectorized float4 reads) over a window of
 *   `working_set_bytes`, repeated `iters` times so the window stays resident
 *   in whichever cache level holds it. Sweeping the working-set size traces a
 *   bandwidth-vs-size curve that plateaus per level (L1 -> L2 -> last-level
 *   cache -> DRAM), with knees at each cache capacity.
 *
 *   mode = "shared":  every block reads the SAME window (one shared buffer).
 *                     Measures the bandwidth a resident buffer delivers to the
 *                     whole device ("global" view). Small windows are replicated
 *                     across each unit's private cache; larger windows are served
 *                     by the shared levels.
 *   mode = "private": each block reads its OWN window (distinct sub-buffers).
 *                     Measures per-unit bandwidth without cross-unit sharing;
 *                     report bandwidth_gbps / num_blocks for the per-unit figure.
 *
 * Reported bandwidth_gbps is the aggregate delivered bandwidth in both modes
 * (bytes = num_blocks * working_set_bytes * iters). per_block_gbps divides by
 * the block count for the per-unit view.
 *
 * Usage:
 *   ./cache_bandwidth [working_set_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_working_set_bytes
 *   BENCH_PARAM_mode             ("shared" | "private")
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

#if defined(__NVCC__) && !__has_include(<hip/hip_runtime.h>)
#include <cuda_runtime.h>
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemset               cudaMemset
#define hipMemGetInfo           cudaMemGetInfo
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetErrorString       cudaGetErrorString
#define hipEvent_t              cudaEvent_t
#define hipEventCreate          cudaEventCreate
#define hipEventDestroy         cudaEventDestroy
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipDeviceProp_t         cudaDeviceProp
#define hipError_t              cudaError_t
#define hipSuccess              cudaSuccess
#define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#else
#include <hip/hip_runtime.h>
#endif

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

static const size_t   DEFAULT_WORKING_SET   = 1ULL * 1024 * 1024;  // 1 MB
static const int      SHARED_BLOCKS_PER_CU      = 8;     // saturate the device (global BW)
static const int      PRIVATE_BLOCKS_PER_CU     = 1;     // 1 block/CU so each owns its L1 (per-unit BW)
static const int      SHARED_THREADS_PER_BLOCK  = 256;   // 8 x 256 = 2048 threads/CU
static const int      PRIVATE_THREADS_PER_BLOCK = 1024;  // max single-block occupancy to saturate one CU
static const size_t   TARGET_TRAFFIC_BYTES  = 8ULL * 1024 * 1024 * 1024;  // 8 GB
static const uint32_t MIN_ITERS             = 3u;
static const uint32_t MAX_ITERS             = 4000000u;
static const size_t   PRIVATE_ALLOC_CAP     = 4ULL * 1024 * 1024 * 1024;  // 4 GB
static const int      NUM_WARMUP            = 2;
static const int      NUM_TIMED             = 3;

// ---------------------------------------------------------------------------
// Streaming-read kernel
// ---------------------------------------------------------------------------

__global__ void bw_read_kernel(const float4 * __restrict__ base,
                               uint32_t  vecs_per_window,
                               uint32_t  windows,
                               uint32_t  iters,
                               int       shared_mode,
                               float    * __restrict__ sink)
{
    const uint32_t w   = shared_mode ? 0u : (blockIdx.x % windows);
    const float4 *win  = base + (size_t)w * vecs_per_window;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t i = threadIdx.x; i < vecs_per_window; i += blockDim.x) {
            float4 v = win[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }

    // Write the accumulator out so the loads are not optimized away.
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    sink[gid] = acc.x + acc.y + acc.z + acc.w;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *env_val;

    long long ws_ll = (long long)DEFAULT_WORKING_SET;
    env_val = getenv("BENCH_PARAM_working_set_bytes");
    if (env_val)            ws_ll = atoll(env_val);
    else if (argc >= 2)     ws_ll = atoll(argv[1]);
    if (ws_ll < 16 || (ws_ll % 16) != 0) {
        fprintf(stderr, "Invalid working_set_bytes (must be >= 16 and a multiple of 16)\n");
        return EXIT_FAILURE;
    }
    size_t working_set_bytes = (size_t)ws_ll;

    const char *mode = "shared";
    env_val = getenv("BENCH_PARAM_mode");
    if (env_val) mode = env_val;
    const int shared_mode = (strcmp(mode, "private") != 0);  // default shared

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    int tpb = shared_mode ? SHARED_THREADS_PER_BLOCK : PRIVATE_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) { int t = atoi(env_val); if (t >= 64 && t <= 1024) tpb = t; }
    int blocks_per_cu = shared_mode ? SHARED_BLOCKS_PER_CU : PRIVATE_BLOCKS_PER_CU;
    env_val = getenv("BENCH_PARAM_blocks_per_cu");
    if (env_val) {
        int bpc = atoi(env_val);
        if (bpc > 0) blocks_per_cu = bpc;
    }
    const int num_blocks = blocks_per_cu * prop.multiProcessorCount;

    size_t free_bytes = 0, total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = free_bytes / 2;  // stay well under free memory

    // Decide how many distinct windows to allocate.
    uint32_t windows;
    size_t   alloc_bytes;
    if (shared_mode) {
        if (working_set_bytes > budget) {
            fprintf(stderr, "Skipping: working_set_bytes %zu exceeds memory budget %zu\n",
                    working_set_bytes, budget);
            printf("{\"type\":\"error\",\"message\":\"working set exceeds memory budget\"}\n");
            return EXIT_FAILURE;
        }
        windows     = 1;
        alloc_bytes = working_set_bytes;
    } else {
        size_t cap = std::min(budget, PRIVATE_ALLOC_CAP);
        size_t max_windows = cap / working_set_bytes;
        if (max_windows < 1) {
            fprintf(stderr, "Skipping: working_set_bytes %zu too large for private mode\n",
                    working_set_bytes);
            printf("{\"type\":\"error\",\"message\":\"working set too large for private mode\"}\n");
            return EXIT_FAILURE;
        }
        windows     = (uint32_t)std::min((size_t)num_blocks, max_windows);
        alloc_bytes = (size_t)windows * working_set_bytes;
    }

    const uint32_t vecs_per_window = (uint32_t)(working_set_bytes / sizeof(float4));

    // Auto-tune iters so each timed launch moves ~TARGET_TRAFFIC_BYTES.
    const size_t per_iter_bytes = (size_t)num_blocks * working_set_bytes;
    uint32_t iters;
    if (per_iter_bytes >= TARGET_TRAFFIC_BYTES) {
        iters = MIN_ITERS;
    } else {
        size_t want = TARGET_TRAFFIC_BYTES / per_iter_bytes;
        iters = (uint32_t)std::min((size_t)MAX_ITERS, std::max((size_t)MIN_ITERS, want));
    }

    fprintf(stderr, "=== cache_bandwidth microbenchmark ===\n");
    fprintf(stderr, "Device          : %s (id=%d, %d CUs)\n",
            prop.name, device_id, prop.multiProcessorCount);
    fprintf(stderr, "Mode            : %s\n", shared_mode ? "shared" : "private");
    fprintf(stderr, "Working set     : %zu bytes (%zu KB)\n",
            working_set_bytes, working_set_bytes / 1024);
    fprintf(stderr, "Grid            : %d blocks (%d/CU) x %d threads\n",
            num_blocks, blocks_per_cu, tpb);
    fprintf(stderr, "Windows / alloc : %u window(s), %zu MB allocated\n",
            windows, alloc_bytes / (1024 * 1024));
    fprintf(stderr, "Iters / launch  : %u\n\n", iters);

    // Allocate and initialize the read buffer + sink.
    float4 *d_buf  = nullptr;
    float  *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, alloc_bytes));
    HIP_CHECK(hipMalloc(&d_sink, (size_t)num_blocks * tpb * sizeof(float)));
    HIP_CHECK(hipMemset(d_buf, 1, alloc_bytes));

    // Warm-up.
    for (int w = 0; w < NUM_WARMUP; ++w) {
        hipLaunchKernelGGL(bw_read_kernel, dim3(num_blocks), dim3(tpb), 0, 0,
                           d_buf, vecs_per_window, windows, iters, shared_mode, d_sink);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed launches: keep the best (lowest time -> peak bandwidth).
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    float best_ms = -1.0f;
    for (int t = 0; t < NUM_TIMED; ++t) {
        HIP_CHECK(hipEventRecord(ev_start, 0));
        hipLaunchKernelGGL(bw_read_kernel, dim3(num_blocks), dim3(tpb), 0, 0,
                           d_buf, vecs_per_window, windows, iters, shared_mode, d_sink);
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        if (best_ms < 0.0f || ms < best_ms) best_ms = ms;
    }

    const double bytes_moved =
        (double)num_blocks * (double)working_set_bytes * (double)iters;
    const double bw_gbps     = bytes_moved / (best_ms * 1e-3) / 1e9;
    const double per_cu_gbps = bw_gbps / (double)prop.multiProcessorCount;

    fprintf(stderr, "WS %zu KB (%s): %.1f GB/s aggregate (%.1f GB/s per CU)\n",
            working_set_bytes / 1024, shared_mode ? "shared" : "private",
            bw_gbps, per_cu_gbps);

    printf("{\"type\":\"kernel\",\"name\":\"bw_read_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"working_set_bytes\":%zu,\"mode\":\"%s\","
           "\"blocks_per_cu\":%d,\"num_blocks\":%d,\"iters\":%u}}\n",
           (double)best_ms, working_set_bytes, shared_mode ? "shared" : "private",
           blocks_per_cu, num_blocks, iters);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"bandwidth_gbps\",\"value\":%.2f},"
           "{\"name\":\"per_cu_gbps\",\"value\":%.4f}]}\n",
           (double)best_ms, bw_gbps, per_cu_gbps);

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipFree(d_sink));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
