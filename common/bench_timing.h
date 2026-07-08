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
 * Overridable via environment:
 *   BENCH_PARAM_num_warmup   (n, default 3)
 *   BENCH_PARAM_num_iters    (m, default 7)
 */

#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "hip_cuda_compat.h"

#define BENCH_DEFAULT_NUM_WARMUP 3
#define BENCH_DEFAULT_NUM_ITERS  7

static inline int bench_env_int(const char *name, int fallback, int min_val)
{
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    int n = atoi(v);
    return n >= min_val ? n : fallback;
}

/* Warmup may legitimately be 0 (e.g. trace collection); iters must be >= 1. */
static inline int bench_num_warmup(void)
{
    return bench_env_int("BENCH_PARAM_num_warmup", BENCH_DEFAULT_NUM_WARMUP, 0);
}

static inline int bench_num_iters(void)
{
    return bench_env_int("BENCH_PARAM_num_iters", BENCH_DEFAULT_NUM_ITERS, 1);
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
 * Warm up `num_warmup` times (each launch synced), then run `num_iters`
 * recorded iterations. `launch` is a callable that enqueues exactly the
 * operation under test on the null stream (a kernel launch or an async
 * copy) and nothing else. Emits one JSON record per iteration plus a
 * per-iteration stderr line, and returns the sum of the recorded GPU
 * event times (ms) for the caller to accumulate into the summary.
 */
template <typename LaunchFn>
static inline double bench_run_loop(const char *name, int num_warmup,
                                    int num_iters, const char *params,
                                    LaunchFn launch)
{
    for (int w = 0; w < num_warmup; ++w) {
        launch();
        BENCH_TIMING_CHECK(hipDeviceSynchronize());
    }

    bench_timer_t timer;
    bench_timer_create(&timer);

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
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
