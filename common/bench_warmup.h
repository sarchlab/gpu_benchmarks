/**
 * bench_warmup.h
 *
 * Standalone clock-warmer / cache-buster for the benchmark harness.
 *
 * Purpose (see the Methodology section of README.md): to match a cold-cache
 * kernel simulator, a timed benchmark kernel must run with (a) the GPU clock
 * already boosted, but (b) its own working set NOT resident in the L2 / last-
 * level cache. This module provides a workload that does exactly that, using a
 * DIFFERENT kernel and DIFFERENT data than the benchmark under test:
 *
 *   - It streams a read-modify-write over a scratch buffer larger than the
 *     last-level cache, repeated until ~BENCH_WARMER_TARGET_MS of sustained
 *     ALU + memory load has run. One pass is only ~0.5-5 ms — far below the
 *     tens of milliseconds GPU DVFS needs to ramp from idle — so the passes
 *     are repeated (mostly inside a single launch via the kernel's rep count)
 *     until the clock has had time to boost. The final pass over the >LLC
 *     buffer evicts the benchmark's working set from the L2/LLC, so the
 *     benchmark kernel that runs immediately afterwards starts cold-cache but
 *     at a boosted clock.
 *   - Because it never touches the benchmark's buffers, it does not warm the
 *     benchmark's data (unlike running the real kernel as a warmup would).
 *
 * The scratch buffer is sized well above the MI300X 256 MB Infinity/MALL last-
 * level cache (and far above any NVIDIA L2), and allocated once on first use.
 *
 * The benchmark's own instruction cache / TLB are not primed by this (different
 * kernel), so the first timed launch pays a one-time I-cache miss the simulator
 * does not model. Every harness repetition is a fresh process and pays it
 * again; it is negligible at kernel timescales.
 */

#ifndef BENCH_WARMUP_H
#define BENCH_WARMUP_H

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "hip_cuda_compat.h"

#ifndef BENCH_WARMUP_CHECK
#define BENCH_WARMUP_CHECK(call)                                              \
    do {                                                                      \
        hipError_t _bw_err = (call);                                          \
        if (_bw_err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error (warmup) at %s:%d — %s\n",            \
                    __FILE__, __LINE__, hipGetErrorString(_bw_err));          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
#endif

/* Scratch bytes: > MI300X 256 MB last-level cache, with headroom. */
#define BENCH_WARMER_BYTES (768ull * 1024 * 1024)

/* Sustained-load target: long enough for DVFS to ramp the clock from idle
 * (tens of ms on every vendor). One warmer pass is ~0.5 ms (MI300X) to ~5 ms
 * (consumer cards), so bench_warmer_run repeats it up to this budget. */
#define BENCH_WARMER_TARGET_MS 100.0

/* Grid-stride read-modify-write: reads and writes every element (evicts the
 * LLC by touching > cache bytes) and does FP work (holds the clock high). The
 * data dependency (out depends on its own prior value) prevents the compiler
 * from eliminating it. */
__global__ void bench_warmer_kernel(float *buf, unsigned long long n, int reps)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (int r = 0; r < reps; ++r) {
        for (unsigned long long i = idx; i < n; i += stride) {
            buf[i] = buf[i] * 1.0000001f + 0.0000001f;
        }
    }
}

static float             *bench_warmer_buf = 0;
static unsigned long long  bench_warmer_n  = 0;

static inline void bench_warmer_init(void)
{
    if (bench_warmer_buf) return;
    bench_warmer_n = BENCH_WARMER_BYTES / sizeof(float);
    BENCH_WARMUP_CHECK(hipMalloc(&bench_warmer_buf,
                                 bench_warmer_n * sizeof(float)));
    BENCH_WARMUP_CHECK(hipMemset(bench_warmer_buf, 1,
                                 bench_warmer_n * sizeof(float)));
}

/* Run the warmer (untimed, synced): repeat the >LLC read-modify-write pass
 * until ~BENCH_WARMER_TARGET_MS of sustained load has executed, so the GPU
 * clock ramps up from idle. The first launch runs a single pass to measure
 * the per-pass time; the remaining budget then runs as one launch with the
 * rep count folded into the kernel (back-to-back passes with no launch gaps
 * — the steadiest load for DVFS). The final pass leaves the benchmark's
 * working set evicted from the last-level cache: boosted clock, cold cache. */
static inline void bench_warmer_run(void)
{
    bench_warmer_init();

    std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    double elapsed_ms = 0.0;
    double pass_ms    = 0.0;
    do {
        int reps = 1;
        if (pass_ms > 0.0) {
            reps = (int)((BENCH_WARMER_TARGET_MS - elapsed_ms) / pass_ms) + 1;
            if (reps < 1) reps = 1;
            if (reps > 1024) reps = 1024; /* hard cap: bounded even if timing misbehaves */
        }
        std::chrono::steady_clock::time_point p0 =
            std::chrono::steady_clock::now();
        bench_warmer_kernel<<<dim3(4096), dim3(256)>>>(bench_warmer_buf,
                                                       bench_warmer_n, reps);
        BENCH_WARMUP_CHECK(hipGetLastError());
        BENCH_WARMUP_CHECK(hipDeviceSynchronize());
        std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
        pass_ms = std::chrono::duration<double, std::milli>(now - p0).count() /
                  reps;
        elapsed_ms =
            std::chrono::duration<double, std::milli>(now - t0).count();
    } while (elapsed_ms < BENCH_WARMER_TARGET_MS);
}

/* BENCH_PARAM_vector_add_warmup — default ON (cold-cache calibration mode).
 * Set to 0/false/no to skip the warmer and time the operation as-is
 * (warm cache, whatever clock state the GPU is in). */
static inline int bench_vector_add_warmup(void)
{
    const char *v = getenv("BENCH_PARAM_vector_add_warmup");
    if (!v || !*v) return 1;
    return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' ||
             v[0] == 'n' || v[0] == 'N');
}

#endif /* BENCH_WARMUP_H */
