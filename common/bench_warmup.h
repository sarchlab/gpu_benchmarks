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
 *     last-level cache, so running it (1) keeps the GPU clock boosted (sustained
 *     ALU + memory work) and (2) evicts the benchmark's working set from the
 *     L2/LLC. The benchmark kernel that runs immediately afterwards therefore
 *     starts cold-cache but at a boosted clock.
 *   - Because it never touches the benchmark's buffers, it does not warm the
 *     benchmark's data (unlike running the real kernel as a warmup would).
 *
 * The scratch buffer is sized well above the MI300X 256 MB Infinity/MALL last-
 * level cache (and far above any NVIDIA L2), and allocated once on first use.
 *
 * The benchmark's own instruction cache / TLB are not primed by this (different
 * kernel), so the first timed launch pays a one-time I-cache miss the simulator
 * does not model — taking the min over the timed iterations discards it.
 */

#ifndef BENCH_WARMUP_H
#define BENCH_WARMUP_H

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

/* Run the warmer once (untimed, synced). One pass over the >LLC buffer evicts
 * the last-level cache; call it before each timed launch to make that launch
 * cold-cache, and repeatedly at startup to boost the clock from idle. */
static inline void bench_warmer_run(void)
{
    bench_warmer_init();
    bench_warmer_kernel<<<dim3(4096), dim3(256)>>>(bench_warmer_buf,
                                                   bench_warmer_n, 1);
    BENCH_WARMUP_CHECK(hipGetLastError());
    BENCH_WARMUP_CHECK(hipDeviceSynchronize());
}

/* BENCH_PARAM_vector_add_warmup — default ON (cold-cache calibration mode).
 * Set to 0/false/no to fall back to warming with the real kernel (warm-cache). */
static inline int bench_vector_add_warmup(void)
{
    const char *v = getenv("BENCH_PARAM_vector_add_warmup");
    if (!v || !*v) return 1;
    return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' ||
             v[0] == 'n' || v[0] == 'N');
}

#endif /* BENCH_WARMUP_H */
