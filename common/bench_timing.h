/**
 * bench_timing.h
 *
 * Shared measurement harness for tier1 microbenchmarks.
 *
 * Contract (single-source with hip_cuda_compat.h):
 *   - Warm up `bench_num_warmup()` times (default 3), each launch synced.
 *   - Loop `bench_num_iters()` times (default 7); every iteration is timed
 *     and emitted individually — no averaging, no min-keeping.
 *   - Each iteration records BOTH host wall-clock (chrono around
 *     launch + sync) and device time (hipEvent pair around the launch).
 *   - The timed bracket contains ONLY the operation under test. Setup
 *     copies, allocation, and verification happen outside it.
 *   - Benchmarks emit raw times and work-descriptor params only; derived
 *     metrics (GFLOPS, GB/s, latency) are computed downstream from the DB.
 *
 * Output protocol (JSON lines on stdout):
 *   {"type":"kernel","name":<op>,"iter":<i>,"time_ms":<gpu event ms>,
 *    "host_time_ms":<chrono ms>,"params":{...}}      one line per iteration
 *   {"type":"summary","total_time_ms":<sum of all gpu event ms>}
 *
 * Three measurement-control parameters (uniform across all benchmarks; these
 * names are deliberately distinct from any benchmark's own workload params
 * such as `iterations` / `num_iterations` / `tsteps`):
 *   BENCH_PARAM_warmup             (n warmup passes, default 3)
 *   BENCH_PARAM_iteration          (m recorded timed iterations, default 7)
 *   BENCH_PARAM_vector_add_warmup  (cold-cache clock warmer on/off, default on)
 *
 * See the Methodology section of README.md for what these measure and why.
 */

#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "hip_cuda_compat.h"
#include "bench_warmup.h"

#define BENCH_DEFAULT_NUM_WARMUP 3
#define BENCH_DEFAULT_NUM_ITERS  7

static inline int bench_env_int(const char *name, int fallback, int min_val)
{
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    int n = atoi(v);
    return n >= min_val ? n : fallback;
}

/* Warmup may legitimately be 0 (e.g. trace collection); iters must be >= 1.
 * Env names are `warmup` / `iteration` (distinct from benchmarks' workload
 * params like `iterations`, `num_iterations`, `tsteps`). */
static inline int bench_num_warmup(void)
{
    return bench_env_int("BENCH_PARAM_warmup", BENCH_DEFAULT_NUM_WARMUP, 0);
}

static inline int bench_num_iters(void)
{
    return bench_env_int("BENCH_PARAM_iteration", BENCH_DEFAULT_NUM_ITERS, 1);
}

/* One iteration's measurement. */
typedef struct {
    double gpu_ms;   /* hipEvent elapsed time around the operation   */
    double host_ms;  /* chrono around operation + full completion    */
} bench_lap_t;

/* Reusable event pair + host clock for per-iteration timing. */
typedef struct {
    hipEvent_t ev_start;
    hipEvent_t ev_stop;
    std::chrono::high_resolution_clock::time_point t0;
} bench_timer_t;

#define BENCH_TIMING_CHECK(call)                                              \
    do {                                                                      \
        hipError_t _bt_err = (call);                                          \
        if (_bt_err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d — %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(_bt_err));                              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static inline void bench_timer_create(bench_timer_t *t)
{
    BENCH_TIMING_CHECK(hipEventCreate(&t->ev_start));
    BENCH_TIMING_CHECK(hipEventCreate(&t->ev_stop));
}

static inline void bench_timer_destroy(bench_timer_t *t)
{
    BENCH_TIMING_CHECK(hipEventDestroy(t->ev_start));
    BENCH_TIMING_CHECK(hipEventDestroy(t->ev_stop));
}

/* Bracket usage:
 *   bench_lap_begin(&t, stream);
 *   <operation under test on `stream`>
 *   lap = bench_lap_end(&t, stream);
 * The gpu time covers exactly the enqueued work between the two event
 * records; the host time additionally covers the event synchronize. */
static inline void bench_lap_begin(bench_timer_t *t, hipStream_t stream)
{
    t->t0 = std::chrono::high_resolution_clock::now();
    BENCH_TIMING_CHECK(hipEventRecord(t->ev_start, stream));
}

static inline bench_lap_t bench_lap_end(bench_timer_t *t, hipStream_t stream)
{
    /* Catch a failed launch inside the bracket (e.g. an invalid launch config)
     * before it is recorded as a bogus time. hipGetLastError clears and returns
     * the error the async launch left pending. */
    BENCH_TIMING_CHECK(hipGetLastError());
    BENCH_TIMING_CHECK(hipEventRecord(t->ev_stop, stream));
    BENCH_TIMING_CHECK(hipEventSynchronize(t->ev_stop));
    bench_lap_t lap;
    lap.host_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::high_resolution_clock::now() - t->t0)
                      .count() /
                  1e6;
    float ms = 0.0f;
    BENCH_TIMING_CHECK(hipEventElapsedTime(&ms, t->ev_start, t->ev_stop));
    lap.gpu_ms = ms;
    return lap;
}

/* Emit one per-iteration kernel record. `params_json` is the inner body of
 * the params object without braces, e.g. "\"num_blocks\":128,\"mode\":\"read\""
 * (may be empty). */
static inline void bench_emit_iter(const char *name, int iter,
                                   const bench_lap_t *lap,
                                   const char *params_json)
{
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"iter\":%d,"
           "\"time_ms\":%.6f,\"host_time_ms\":%.6f,\"params\":{%s}}\n",
           name, iter, lap->gpu_ms, lap->host_ms,
           params_json ? params_json : "");
}

/* Emit the final summary: the sum of every emitted iteration's gpu time. */
static inline void bench_emit_summary(double total_gpu_ms)
{
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f}\n", total_gpu_ms);
}

/**
 * bench_run_loop — the measurement contract in one place.
 *
 * With BENCH_PARAM_vector_add_warmup on (default), this measures the operation
 * cold-cache but at a boosted clock (matching a cold-cache kernel simulator):
 * the warmup runs the vector-add clock warmer `num_warmup` times (boosting the
 * clock without touching the benchmark's data), and before each timed iteration
 * the warmer runs once more to evict the last-level cache and hold the clock
 * high — so every timed launch starts cold. Take the min over iterations for
 * the least-perturbed sample.
 *
 * With it off, it falls back to warm-cache mode: warm up by running the real
 * operation `num_warmup` times (each synced), then time it warm.
 *
 * `launch` enqueues exactly the operation under test on the null stream.
 * Emits one JSON record per iteration and returns the sum of GPU event times.
 */
template <typename LaunchFn>
static inline double bench_run_loop(const char *name, int num_warmup,
                                    int num_iters, const char *params,
                                    LaunchFn launch)
{
    const int cold = bench_vector_add_warmup();

    for (int w = 0; w < num_warmup; ++w) {
        if (cold) {
            bench_warmer_run();          /* boost clock; leave benchmark data cold */
        } else {
            launch();                    /* legacy warm-cache warmup */
            BENCH_TIMING_CHECK(hipDeviceSynchronize());
        }
    }

    bench_timer_t timer;
    bench_timer_create(&timer);

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        if (cold) bench_warmer_run();    /* evict LLC + hold clock -> cold launch */
        bench_lap_begin(&timer, 0);
        launch();
        bench_lap_t lap = bench_lap_end(&timer, 0);

        bench_emit_iter(name, i, &lap, params);
        total_gpu_ms += lap.gpu_ms;
        fprintf(stderr, "%s iter %d: gpu = %.3f ms, host = %.3f ms\n",
                name, i, lap.gpu_ms, lap.host_ms);
    }

    bench_timer_destroy(&timer);
    return total_gpu_ms;
}

/* Multi-kernel / iterative benchmarks drive their own pass loop with
 * bench_time_op. Call this at the start of each recorded pass (before the timed
 * launches) so the pass runs cold-cache at a boosted clock, matching bench_run_loop.
 * No-op when vector-add warmup is disabled. */
static inline void bench_pass_warm(void)
{
    if (bench_vector_add_warmup()) bench_warmer_run();
}

/**
 * Multi-kernel workloads (tier2): run the whole workload num_iters times;
 * within each pass, bracket every kernel launch with bench_time_op and
 * accumulate into that kernel's per-pass bench_lap_t via bench_lap_add,
 * then emit one record per kernel per pass with bench_emit_iter. The
 * summary stays the sum of every emitted record's gpu time.
 */
template <typename LaunchFn>
static inline bench_lap_t bench_time_op(bench_timer_t *t, LaunchFn op)
{
    bench_lap_begin(t, 0);
    op();
    return bench_lap_end(t, 0);
}

static inline void bench_lap_reset(bench_lap_t *acc)
{
    acc->gpu_ms  = 0.0;
    acc->host_ms = 0.0;
}

static inline void bench_lap_add(bench_lap_t *acc, bench_lap_t lap)
{
    acc->gpu_ms  += lap.gpu_ms;
    acc->host_ms += lap.host_ms;
}

/** bench_run_memcpy — shared runner for memcpy-based benchmarks
 *  (memory_bandwidth, pcie_bandwidth unidirectional tests). */
static inline double bench_run_memcpy(const char *name, void *dst,
                                      const void *src, size_t bytes,
                                      hipMemcpyKind kind, int num_warmup,
                                      int num_iters, const char *params)
{
    return bench_run_loop(name, num_warmup, num_iters, params, [&]() {
        BENCH_TIMING_CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
    });
}

#endif /* BENCH_TIMING_H */
