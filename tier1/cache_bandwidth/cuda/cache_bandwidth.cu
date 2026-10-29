/**
 * cache_bandwidth.cu
 *
 * Native CUDA version of the cache_bandwidth microbenchmark (rename-only port of ../hip/).
 * Compiles with nvcc for NVIDIA GPUs.
 *
 * Microbenchmark for repeated-window logical read bandwidth.
 * Each block streams float4 reads over working_set_bytes, repeated iters times.
 * All passes, including the first fill, are timed. Window-size sweeps study
 * memory-hierarchy behavior; timing alone does not establish cache residency.
 *
 *   mode = "shared":  every block reads the SAME window (one shared buffer).
 *                     Measures the bandwidth a resident buffer delivers to the
 *                     whole device ("global" view). Small windows are replicated
 *                     across each unit's private cache; larger windows are served
 *                     by the shared levels.
 *   mode = "private": each block reads its OWN window (distinct sub-buffers).
 *                     Reduce the grid when the allocation cap cannot fit all
 *                     requested windows. Distinct blocks do not imply CU placement.
 *
 * Usage:
 *   ./cache_bandwidth [working_set_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_working_set_bytes
 *   BENCH_PARAM_mode             ("shared" | "private")
 *
 * Output: one {"type":"kernel",...} JSON line per timed iteration carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all iterations' GPU event times. Bandwidth is
 * derived downstream: bytes = num_blocks * working_set_bytes * iters (all in
 * the record params) over time_ms.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

#include "../../../common_cuda/bench_timing.h"

#define CUDA_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const size_t   DEFAULT_WORKING_SET   = 1ULL * 1024 * 1024;  // 1 MB
static const int      SHARED_BLOCKS_PER_CU      = 8;     // saturate the device (global BW)
static const int      PRIVATE_BLOCKS_PER_CU     = 1;     // request one distinct-window block per CU
static const int      SHARED_THREADS_PER_BLOCK  = 256;   // 8 x 256 = 2048 threads/CU
static const int      PRIVATE_THREADS_PER_BLOCK = 1024;  // max single-block occupancy to saturate one CU
static const size_t   TARGET_TRAFFIC_BYTES  = 8ULL * 1024 * 1024 * 1024;  // 8 GB
static const uint32_t MIN_ITERS             = 3u;
static const uint32_t MAX_ITERS             = 4000000u;
static const size_t   PRIVATE_ALLOC_CAP     = 4ULL * 1024 * 1024 * 1024;  // 4 GB

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
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int tpb = shared_mode ? SHARED_THREADS_PER_BLOCK : PRIVATE_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) { int t = atoi(env_val); if (t >= 64 && t <= 1024) tpb = t; }
    int blocks_per_cu = shared_mode ? SHARED_BLOCKS_PER_CU : PRIVATE_BLOCKS_PER_CU;
    env_val = getenv("BENCH_PARAM_blocks_per_cu");
    if (env_val) {
        int bpc = atoi(env_val);
        if (bpc > 0) blocks_per_cu = bpc;
    }
    if (prop.multiProcessorCount <= 0 || prop.maxGridSize[0] <= 0 ||
        tpb > prop.maxThreadsPerBlock) {
        fprintf(stderr, "Unsupported device launch limits\n");
        return EXIT_FAILURE;
    }
    const uint64_t requested_num_blocks =
        (uint64_t)blocks_per_cu * (uint64_t)prop.multiProcessorCount;
    // The last loop increment must not wrap its uint32_t index.
    if (working_set_bytes / sizeof(float4) > UINT32_MAX - (uint32_t)(tpb - 1)) {
        fprintf(stderr, "Working-set traversal exceeds uint32_t\n");
        return EXIT_FAILURE;
    }

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = free_bytes / 2;  // stay well under free memory

    // Decide the actual grid and distinct-window allocation together.
    uint64_t effective_blocks = requested_num_blocks;
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
        effective_blocks = std::min(requested_num_blocks, (uint64_t)max_windows);
        // Also checked below before launch; never truncate the window count.
        if (effective_blocks > UINT32_MAX) {
            fprintf(stderr, "Private window count exceeds uint32_t\n");
            return EXIT_FAILURE;
        }
        windows     = (uint32_t)effective_blocks;
        alloc_bytes = (size_t)windows * working_set_bytes;
    }

    // Keep grid, sink indices and byte accounting representable.
    if (effective_blocks > (uint64_t)prop.maxGridSize[0] ||
        effective_blocks > ((uint64_t)UINT32_MAX + 1) / (uint32_t)tpb ||
        effective_blocks > SIZE_MAX / working_set_bytes ||
        effective_blocks > SIZE_MAX / ((size_t)tpb * sizeof(float))) {
        fprintf(stderr, "Effective launch geometry exceeds supported indices or sizes\n");
        return EXIT_FAILURE;
    }
    const uint32_t num_blocks = (uint32_t)effective_blocks;
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

    if (per_iter_bytes > UINT64_MAX / iters) {
        fprintf(stderr, "Launch byte count exceeds uint64_t\n");
        return EXIT_FAILURE;
    }
    const uint64_t bytes_read = (uint64_t)per_iter_bytes * iters;

    fprintf(stderr, "=== cache_bandwidth microbenchmark ===\n");
    fprintf(stderr, "Device          : %s (id=%d, %d CUs)\n",
            prop.name, device_id, prop.multiProcessorCount);
    fprintf(stderr, "Mode            : %s\n", shared_mode ? "shared" : "private");
    fprintf(stderr, "Working set     : %zu bytes (%zu KB)\n",
            working_set_bytes, working_set_bytes / 1024);
    fprintf(stderr, "Requested grid  : %llu blocks (%d per reported CU)\n",
            (unsigned long long)requested_num_blocks, blocks_per_cu);
    fprintf(stderr, "Effective grid  : %u blocks x %d threads\n", num_blocks, tpb);
    fprintf(stderr, "Windows / alloc : %u window(s), %zu MB allocated\n",
            windows, alloc_bytes / (1024 * 1024));
    fprintf(stderr, "Iters / launch  : %u\n\n", iters);

    // Allocate and initialize the read buffer + sink.
    float4 *d_buf  = nullptr;
    float  *d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, alloc_bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, (size_t)num_blocks * tpb * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf, 1, alloc_bytes));

    char params[768];
    snprintf(params, sizeof(params),
             "\"working_set_bytes\":%zu,\"mode\":\"%s\","
             "\"blocks_per_cu\":%d,\"requested_num_blocks\":%llu,"
             "\"num_blocks\":%u,\"threads_per_block\":%d,\"windows\":%u,"
             "\"allocation_bytes\":%zu,\"iters\":%u,\"bytes_per_iter\":%zu,"
             "\"bytes_read\":%llu,\"launch_policy\":\"%s\"",
             working_set_bytes, shared_mode ? "shared" : "private",
             blocks_per_cu, (unsigned long long)requested_num_blocks,
             num_blocks, tpb, windows, alloc_bytes, iters, per_iter_bytes,
             (unsigned long long)bytes_read,
             shared_mode ? "shared_window_v1" : "distinct_windows_v1");

    double total_gpu_ms =
        bench_run_once("bw_read_kernel", params, [&]() {
            bw_read_kernel<<<dim3(num_blocks), dim3(tpb)>>>(
                d_buf, vecs_per_window, windows, iters, shared_mode, d_sink);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_sink));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
