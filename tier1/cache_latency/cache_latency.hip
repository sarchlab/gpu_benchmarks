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
 *   Three tests vary the array size to target different levels of the
 *   GPU memory hierarchy:
 *
 *     Test         Array size   Intended cache level
 *     ──────────── ──────────── ────────────────────
 *     L1           16 KB        L1 data cache
 *     L2            2 MB        L2 cache (beyond L1)
 *     Global mem   64 MB        DRAM (beyond L2)
 *
 *   The chain visits every element in a uniformly random order (Knuth
 *   shuffle), which defeats hardware stream-prefetchers.
 *
 * Output:
 *   Latency in nanoseconds per memory access for each cache level.
 *
 * Usage:
 *   ./cache_latency
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include <hip/hip_runtime.h>

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
// Constants
// ---------------------------------------------------------------------------

// Array sizes for each cache-level test.
// Use powers-of-two for clean alignment.
static const size_t L1_ARRAY_BYTES   = 16ULL  * 1024;           //  16 KB
static const size_t L2_ARRAY_BYTES   = 2ULL   * 1024 * 1024;    //   2 MB
static const size_t GLOB_ARRAY_BYTES = 64ULL  * 1024 * 1024;    //  64 MB

// Number of pointer-chasing steps per kernel launch.
static const uint32_t L1_ACCESSES   = 10000000u;  // 10 M steps
static const uint32_t L2_ACCESSES   =  2000000u;  //  2 M steps
static const uint32_t GLOB_ACCESSES =   300000u;  // 300 K steps

// Timing iterations (averaged to reduce variability).
static const int NUM_ITERS = 5;

// RNG seed (reproducible traversal order).
static const uint32_t RNG_SEED = 42u;

// ---------------------------------------------------------------------------
// Pointer-chasing kernel
// ---------------------------------------------------------------------------

/**
 * pointer_chase_kernel — single-threaded pointer-chasing latency measurement.
 *
 * Walks the index chain starting at @p start_idx for @p num_accesses steps.
 * The true data dependency between iterations (idx = arr[idx]) forces the
 * hardware to fully resolve each load before issuing the next, making the
 * total runtime proportional to per-access latency.
 *
 * @param arr          Device array: arr[i] is the next index after i.
 * @param start_idx    First index in the chain.
 * @param num_accesses Total steps to take.
 * @param result       Output location (prevents dead-code elimination).
 */
__global__ void pointer_chase_kernel(
    const uint32_t * __restrict__ arr,
    uint32_t  start_idx,
    uint32_t  num_accesses,
    uint32_t *result)
{
    // Only a single thread (block 0, thread 0) participates.
    // Using one thread ensures we measure latency, not throughput.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t idx = start_idx;

    for (uint32_t i = 0; i < num_accesses; ++i) {
        // Dependent load: next address is contained in the current element.
        // This creates a serial dependency chain that hardware cannot pipeline.
        idx = arr[idx];
    }

    // Write the final index so the compiler cannot eliminate the loop.
    *result = idx;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/**
 * build_chain — build a random Hamiltonian pointer-chasing cycle.
 *
 * Constructs a single cycle of length N that visits every element exactly
 * once before returning to the start.  Random order defeats prefetchers.
 *
 * @param arr       Output host array of N uint32_t indices.
 * @param N         Number of elements.
 * @param start_idx (Out) The first index in the cycle.
 * @param seed      RNG seed for reproducibility.
 */
static void build_chain(uint32_t       *arr,
                        uint32_t        N,
                        uint32_t       &start_idx,
                        uint32_t        seed = RNG_SEED)
{
    // Build a random permutation of [0 .. N-1].
    std::mt19937 rng(seed);
    std::vector<uint32_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Wire the cycle: element at position perm[i] points to perm[i+1 % N].
    // This guarantees a single cycle of length N (Hamiltonian cycle).
    for (uint32_t i = 0; i < N; ++i) {
        arr[perm[i]] = perm[(i + 1u) % N];
    }

    start_idx = perm[0];
}

// ---------------------------------------------------------------------------
// Per-test runner
// ---------------------------------------------------------------------------

/**
 * Descriptor for a single cache-level test.
 */
struct TestConfig {
    const char *name;           //< Human-readable label
    size_t      array_bytes;    //< Array size (determines targeted cache level)
    uint32_t    num_accesses;   //< Pointer-chasing steps per kernel launch
    int         num_iters;      //< Number of timed kernel launches
};

/**
 * run_latency_test — set up, warm up, time, and tear down one latency test.
 *
 * @param cfg  Test configuration.
 * @return     Average latency in nanoseconds per access.
 */
static double run_latency_test(const TestConfig &cfg)
{
    const size_t   N     = cfg.array_bytes / sizeof(uint32_t);
    const uint32_t N_u32 = static_cast<uint32_t>(N);

    printf("  [%s] array = %zu KB, accesses = %u, iters = %d\n",
           cfg.name,
           cfg.array_bytes / 1024,
           cfg.num_accesses,
           cfg.num_iters);
    fflush(stdout);

    // --- Build chain on host ---
    std::vector<uint32_t> h_arr(N);
    uint32_t start_idx = 0;
    build_chain(h_arr.data(), N_u32, start_idx);

    // --- Allocate device memory ---
    uint32_t *d_arr    = nullptr;
    uint32_t *d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_arr,    cfg.array_bytes));
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint32_t)));

    // --- Upload chain ---
    HIP_CHECK(hipMemcpy(d_arr, h_arr.data(), cfg.array_bytes,
                        hipMemcpyHostToDevice));

    // --- Warm-up: prime the cache and hide first-launch overheads ---
    hipLaunchKernelGGL(pointer_chase_kernel, dim3(1), dim3(1), 0, 0,
                       d_arr, start_idx, cfg.num_accesses, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Timed runs ---
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start, 0));
    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        hipLaunchKernelGGL(pointer_chase_kernel, dim3(1), dim3(1), 0, 0,
                           d_arr, start_idx, cfg.num_accesses, d_result);
    }
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));

    // Latency (ns) = total_time_ns / total_accesses
    const double total_accesses =
        static_cast<double>(cfg.num_accesses) * cfg.num_iters;
    const double latency_ns = static_cast<double>(total_ms) * 1e6 / total_accesses;

    // --- Cleanup ---
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_arr));
    HIP_CHECK(hipFree(d_result));

    return latency_ns;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void)
{
    // Print device information.
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    printf("=== cache_latency microbenchmark ===\n");
    printf("Device : %s (id=%d)\n", prop.name, device_id);
    printf("Method : single-thread pointer chasing (random permutation)\n");
    printf("Each test: one warm-up + %d timed kernel launch(es)\n\n",
           NUM_ITERS);

    // Define the three cache-level tests.
    const TestConfig tests[] = {
        { "L1 cache  ",  L1_ARRAY_BYTES,   L1_ACCESSES,   NUM_ITERS },
        { "L2 cache  ",  L2_ARRAY_BYTES,   L2_ACCESSES,   NUM_ITERS },
        { "Global mem",  GLOB_ARRAY_BYTES, GLOB_ACCESSES, NUM_ITERS },
    };
    const int NUM_TESTS = static_cast<int>(sizeof(tests) / sizeof(tests[0]));

    printf("Running tests...\n");
    double results[3] = {};

    for (int t = 0; t < NUM_TESTS; ++t) {
        results[t] = run_latency_test(tests[t]);
    }

    // Print summary.
    printf("\n%-14s  %10s  %10s\n",
           "Cache level", "Array size", "Latency");
    printf("%-14s  %10s  %10s\n",
           "──────────", "──────────", "──────────");
    for (int t = 0; t < NUM_TESTS; ++t) {
        const char *unit;
        double sz;
        if (tests[t].array_bytes >= 1024 * 1024) {
            sz   = static_cast<double>(tests[t].array_bytes) / (1024.0 * 1024.0);
            unit = "MB";
        } else {
            sz   = static_cast<double>(tests[t].array_bytes) / 1024.0;
            unit = "KB";
        }
        printf("%-14s  %7.0f %-2s  %8.2f ns\n",
               tests[t].name, sz, unit, results[t]);
    }

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
