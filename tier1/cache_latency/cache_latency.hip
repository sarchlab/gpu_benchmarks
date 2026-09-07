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
 *   Chase nodes use a configured byte spacing (historically cacheline_bytes).
 *   Consecutive accesses visit distinct nodes. Physical line/sector sizes and
 *   the effective cache footprint must be qualified on the target.
 *
 *   Entry policy is primed: optional vector-add clock warmer, then an
 *   untimed pointer-chase primer, then one timed chase. The primer is capped
 *   at 256 MiB of configured node spacing. This is an explicit exception to
 *   the suite's default cold-cache entry. Keep the primer during tracing.
 *   Priming targets resident-cache measurements; actual hit levels depend on
 *   capacity, load policy and launch placement. A separate primer does not
 *   guarantee L1 residency or eliminate all timed misses (see README).
 *   Timed accesses = clamp(measure_laps * nodes, MIN, MAX).
 *
 * Usage:
 *   ./cache_latency [array_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_array_bytes
 *   BENCH_PARAM_cacheline_bytes   (default 64)
 *   BENCH_PARAM_measure_laps      (default 4 requested timed laps after priming)
 *   BENCH_PARAM_num_accesses      (overrides the timed-access count)
 *   BENCH_PARAM_rng_seed
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
static const uint32_t DEFAULT_CACHELINE_BYTES = 64u;       // configured node spacing; qualify physical lines per target
static const uint32_t DEFAULT_MEASURE_LAPS    = 4u;        // requested timed laps after the untimed primer
static const size_t   WARMUP_CAP_BYTES        = 256ULL * 1024 * 1024; // bound primer cost; not a discovered cache size
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
 * Node indices are {0, stride, 2*stride, ...}, stride = cacheline_bytes/4.
 * The configured spacing spans the array; actual line/sector sharing and
 * cache residency depend on the target and compiled load policy.
 *
 *   arr      : host index array (array_bytes/4 elements; only node slots written)
 *   M        : number of configured chase nodes
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

    /* Pointer-chase nodes at the configured byte spacing. */
    const uint32_t stride = cacheline_bytes / (uint32_t)sizeof(uint32_t);
    if (stride == 0 || N < (size_t)stride * 2) {
        fprintf(stderr, "array_bytes (%zu) too small for cacheline_bytes (%u); "
                        "need >= 2 cachelines\n", array_bytes, cacheline_bytes);
        return EXIT_FAILURE;
    }
    const uint32_t M = (uint32_t)(N / stride);   // number of configured chase nodes

    /* Prime one lap when within the fixed cap; bound setup cost for larger
     * chains. This access count does not establish a target's cache capacity. */
    const uint64_t warm_cap = (uint64_t)(WARMUP_CAP_BYTES / cacheline_bytes);
    uint64_t num_warmup = ((uint64_t)M < warm_cap) ? (uint64_t)M : warm_cap;
    if (num_warmup < 1u) num_warmup = 1u;

    /* Timed accesses = measure_laps steady-state laps, clamped: a floor so tiny
     * resident sizes still get enough samples, a ceiling so the DRAM end stays
     * bounded. Primer work is excluded; per-launch L1 fill or capacity misses
     * can still occur in the timed chase. Override wins. */
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
    fprintf(stderr, "Node spacing   : %u B  (stride %u elems)\n", cacheline_bytes, stride);
    fprintf(stderr, "Nodes          : %u\n", M);
    fprintf(stderr, "Primer         : %llu accesses (untimed)\n",
            (unsigned long long)num_warmup);
    fprintf(stderr, "Timed chase    : %u laps -> %llu accesses\n",
            measure_laps, (unsigned long long)num_timed);
    fprintf(stderr, "RNG seed       : %u\n", rng_seed);
    fprintf(stderr, "Entry policy   : optional warmer -> untimed primer -> timed chase\n\n");

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

    /* Resident-cache entry exception: condition clocks/cache first, then
     * prime our chain. No cache-evicting work may run after this primer. */
    bench_pass_warm();

    /* One untimed primer, retained even when the clock warmer is disabled
     * (including trace collection). Large chains receive a capped prefix. */
    pointer_chase_kernel<<<dim3(1), dim3(1)>>>(d_arr, start_idx, num_warmup, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    char params[320];
    snprintf(params, sizeof(params),
             "\"array_bytes\":%zu,\"num_accesses\":%llu,"
             "\"warmup_accesses\":%llu,\"measure_laps\":%u,"
             "\"cacheline_bytes\":%u,\"rng_seed\":%u,"
             "\"cache_entry\":\"primed\",\"primer_launches\":1",
             array_bytes, (unsigned long long)num_timed,
             (unsigned long long)num_warmup, measure_laps,
             cacheline_bytes, rng_seed);

    /* Use the existing event timer directly so bench_run_once cannot insert
     * another warmer between the primer and the measured launch. */
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t lap = bench_time_op(&timer, [&]() {
        pointer_chase_kernel<<<dim3(1), dim3(1)>>>(d_arr, start_idx, num_timed, d_result);
    });
    bench_timer_destroy(&timer);
    bench_emit_iter("pointer_chase_kernel", 0, &lap, params);
    double total_gpu_ms = lap.gpu_ms;
    fprintf(stderr, "pointer_chase_kernel: gpu = %.3f ms, host = %.3f ms\n",
            lap.gpu_ms, lap.host_ms);

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    /* Cleanup. */
    HIP_CHECK(hipFree(d_arr));
    HIP_CHECK(hipFree(d_result));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
