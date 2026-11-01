/**
 * bench_timing.h
 *
 * Shared measurement harness for tier1/tier2 benchmarks.
 *
 * Contract (single-source with hip_cuda_compat.h):
 *   - Each process runs the operation under test EXACTLY ONCE and times it.
 *     Repetition across fresh processes is the harness's job (gpubench --reps);
 *     there is no in-program warmup or iteration loop.
 *   - When BENCH_PARAM_vector_add_warmup is on (default), the vector-add warmer
 *     runs immediately before the timed run — repeated to a ~100 ms
 *     sustained-load target so the GPU clock ramps up from idle, with the
 *     final pass evicting the last-level cache — so the operation is measured
 *     cold-cache at a boosted clock (matching a cold-cache kernel simulator).
 *     With it off, the operation is timed as-is.
 *   - The single timed run records BOTH host wall-clock (chrono around
 *     launch + sync) and device time (hipEvent pair around the launch).
 *   - The timed bracket contains ONLY the operation under test. Setup copies,
 *     allocation, and verification happen outside it.
 *   - Benchmarks emit raw times and work-descriptor params only; derived
 *     metrics (GFLOPS, GB/s, latency) are computed downstream from the DB.
 *
 * Output protocol (JSON lines on stdout):
 *   {"type":"kernel","name":<op>,"iter":0,"time_ms":<gpu event ms>,
 *    "host_time_ms":<chrono ms>,"params":{...}}      one line per kernel
 *   {"type":"summary","total_time_ms":<sum of all gpu event ms>}
 *
 * One measurement-control parameter (uniform across all benchmarks):
 *   BENCH_PARAM_vector_add_warmup  (cold-cache clock warmer on/off, default on)
 *
 * See the Methodology section of README.md for what this measures and why.
 */

#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "hip_cuda_compat.h"
#include "bench_warmup.h"

/* One run's measurement. */
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
 * bench_run_once — measure a single-kernel benchmark: one timed run.
 *
 * With BENCH_PARAM_vector_add_warmup on (default), the vector-add warmer runs
 * first (repeated to a ~100 ms sustained-load target) to boost the GPU clock,
 * its final pass evicting the last-level cache, so the timed launch runs
 * cold-cache at a boosted clock (matching a cold-cache kernel simulator).
 * With it off, the launch is timed as-is. Repetition across processes is the
 * harness's job (gpubench --reps); this runs `launch` exactly once.
 *
 * `launch` enqueues exactly the operation under test on the null stream.
 * Emits one kernel record and returns its GPU event time (ms).
 */
template <typename LaunchFn>
static inline double bench_run_once(const char *name, const char *params,
                                    LaunchFn launch)
{
    if (bench_vector_add_warmup()) bench_warmer_run();  /* boost clock + cold cache */

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_begin(&timer, 0);
    launch();
    bench_lap_t lap = bench_lap_end(&timer, 0);
    bench_timer_destroy(&timer);

    bench_emit_iter(name, 0, &lap, params);
    fprintf(stderr, "%s: gpu = %.3f ms, host = %.3f ms\n",
            name, lap.gpu_ms, lap.host_ms);
    return lap.gpu_ms;
}

/* Multi-kernel / iterative benchmarks drive their own kernel launches with
 * bench_time_op. Call this once at the start of the (single) timed run, after
 * per-pass data (re)initialization and before the first timed launch, so the
 * run is cold-cache at a boosted clock. No-op when vector-add warmup is off. */
static inline void bench_pass_warm(void)
{
    if (bench_vector_add_warmup()) bench_warmer_run();
}

/**
 * Multi-kernel workloads (tier2): run the whole workload once; bracket every
 * kernel launch with bench_time_op and accumulate into that kernel's
 * bench_lap_t via bench_lap_add, then emit one record per kernel with
 * bench_emit_iter (iter 0). The summary is the sum of every emitted record's
 * gpu time.
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
                                      hipMemcpyKind kind, const char *params)
{
    return bench_run_once(name, params, [&]() {
        BENCH_TIMING_CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
    });
}

#endif /* BENCH_TIMING_H */
