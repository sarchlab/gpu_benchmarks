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
 *   the whole array (cache footprint == array_bytes).
 *
 *   To avoid biasing the fast levels, the COLD fill lap is run UNTIMED and
 *   discarded: an untimed warm-up walks one lap (capped at the largest cache
 *   so the DRAM end doesn't pay a huge fill), which both loads the kernel
 *   code object and makes the working set resident; only steady-state laps
 *   are timed. Timed accesses = clamp(measure_laps * lines, MIN, MAX), so
 *   resident sizes measure pure hits (L1/L2/MALL), sizes past the caches
 *   measure all-miss (DRAM) latency, and the count is bounded so every point
 *   runs in seconds regardless of array size.
 *
 * Usage:
 *   ./cache_latency [array_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_array_bytes
 *   BENCH_PARAM_cacheline_bytes   (default 64)
 *   BENCH_PARAM_measure_laps      (default 4 timed laps of steady-state hits)
 *   BENCH_PARAM_num_accesses      (overrides the timed-access count)
 *   BENCH_PARAM_rng_seed
 *   BENCH_PARAM_num_warmup        (untimed cold-fill launches, default 3)
 *   BENCH_PARAM_num_iters         (timed launches, each recorded, default 7)
 *
 * Output: one {"type":"kernel",...} JSON line per timed iteration carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all iterations' GPU event times. Latency per
 * access is derived downstream as time_ms*1e6 / num_accesses (both are in
 * the record).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include "../../common/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const size_t   DEFAULT_ARRAY_BYTES     = 2ULL * 1024 * 1024;  // 2 MB
static const uint32_t DEFAULT_CACHELINE_BYTES = 64u;       // gfx942/MI300X line; one chase node per line
static const uint32_t DEFAULT_MEASURE_LAPS    = 4u;        // timed laps of steady-state hits (the cold lap is untimed)
static const size_t   WARMUP_CAP_BYTES        = 256ULL * 1024 * 1024; // cap the untimed fill at the largest cache (MALL); DRAM needs no fill
static const uint64_t MIN_TIMED_ACCESSES      = 131072u;   // floor so tiny resident sizes still get enough samples
static const uint64_t MAX_TIMED_ACCESSES      = 16000000u; // ceiling so the DRAM end stays bounded (~seconds, fits the harness timeout)
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

    uint32_t measure_laps = DEFAULT_MEASURE_LAPS;
    env_val = getenv("BENCH_PARAM_measure_laps");
    if (env_val) measure_laps = (uint32_t)atoi(env_val);

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

    /* Untimed warm-up = one lap that makes the set resident (and loads the code
     * object), capped at the largest cache so the DRAM end skips a huge fill. */
    const uint64_t warm_cap = (uint64_t)(WARMUP_CAP_BYTES / cacheline_bytes);
    uint64_t num_warmup = ((uint64_t)M < warm_cap) ? (uint64_t)M : warm_cap;
    if (num_warmup < 1u) num_warmup = 1u;

    /* Timed accesses = measure_laps steady-state laps, clamped: a floor so tiny
     * resident sizes still get enough samples, a ceiling so the DRAM end stays
     * bounded. The cold fill is excluded (run untimed above), so these measure
     * pure hits for resident sizes and pure misses past the caches. Override wins. */
    uint64_t num_timed = (uint64_t)measure_laps * (uint64_t)M;
    if (num_timed < MIN_TIMED_ACCESSES) num_timed = MIN_TIMED_ACCESSES;
    if (num_timed > MAX_TIMED_ACCESSES) num_timed = MAX_TIMED_ACCESSES;
    env_val = getenv("BENCH_PARAM_num_accesses");
    if (env_val) num_timed = strtoull(env_val, nullptr, 10);

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
    fprintf(stderr, "Warm-up (cold) : %llu accesses (untimed)\n",
            (unsigned long long)num_warmup);
    fprintf(stderr, "Timed (steady) : %u laps -> %llu accesses\n",
            measure_laps, (unsigned long long)num_timed);
    fprintf(stderr, "RNG seed       : %u\n", rng_seed);
    fprintf(stderr, "Method         : cacheline-stride pointer chase, cold lap discarded\n\n");

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

    /* Untimed warm-up launches: the first makes the working set resident (so
     * timed runs measure steady-state hits, not the cold fill) and loads the
     * kernel code object; later ones are cheap steady-state laps. Capped at
     * the largest cache, so the DRAM end pays only a partial fill it doesn't
     * need anyway. Excluded from timing. */
    const int warmup_launches = bench_num_warmup();
    const int num_iters       = bench_num_iters();
    for (int w = 0; w < warmup_launches; ++w) {
        pointer_chase_kernel<<<dim3(1), dim3(1)>>>(d_arr, start_idx, num_warmup, d_result);
        HIP_CHECK(hipDeviceSynchronize());
    }

    char params[256];
    snprintf(params, sizeof(params),
             "\"array_bytes\":%zu,\"num_accesses\":%llu,"
             "\"warmup_accesses\":%llu,\"measure_laps\":%u,"
             "\"cacheline_bytes\":%u,\"rng_seed\":%u",
             array_bytes, (unsigned long long)num_timed,
             (unsigned long long)num_warmup, measure_laps,
             cacheline_bytes, rng_seed);

    /* Warmup is done separately above (it chases num_warmup cold-fill accesses,
     * a different count than the timed num_timed), so run the timed loop with
     * num_warmup=0. Each timed launch chases num_timed accesses and is recorded
     * individually; the working set stays resident so every iteration measures
     * the same steady state. */
    double total_gpu_ms =
        bench_run_loop("pointer_chase_kernel", 0, num_iters, params, [&]() {
            pointer_chase_kernel<<<dim3(1), dim3(1)>>>(d_arr, start_idx, num_timed, d_result);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    /* Cleanup. */
    HIP_CHECK(hipFree(d_arr));
    HIP_CHECK(hipFree(d_result));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
