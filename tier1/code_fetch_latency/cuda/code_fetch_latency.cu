/**
 * code_fetch_latency.cu
 *
 * Native CUDA version of the code_fetch_latency microbenchmark (rename-only port of ../hip/).
 * Compiles with nvcc for NVIDIA GPUs.
 *
 * Microbenchmark for the cost of fetching kernel code that is not yet
 * resident in the instruction cache.
 *
 * Methodology:
 *   Every thread runs one dependent chain of total_ops fused multiply-adds,
 *   x = fmaf(x, a, b), on register values. The chain is laid out as a
 *   straight-line body of body_ops FMAs inside a loop of
 *   reps = total_ops / body_ops iterations. Executed work is therefore the
 *   same for every body_ops; only the static code footprint changes. One
 *   kernel exists per supported body size, so a launch fetches only the code
 *   of the size under test.
 *
 *   Entry policy: optional vector-add clock/cache warmer, then
 *   prime_launches untimed launches of the same kernel, then one timed
 *   launch. With prime_launches=0 the timed launch is the first launch of
 *   that kernel in the process (the suite's default cold entry). With
 *   prime_launches=1 its code was fetched once already: an explicit
 *   resident-code exception to cold entry. Keep the primer during tracing.
 *   A primer runs the same kernel with the same reps but stores to its own
 *   buffer, so the timed launch's output is untouched in both entries. The
 *   difference between the two entries at one body size is the cost of
 *   fetching that kernel's code under this protocol (see README).
 *
 * Usage:
 *   ./code_fetch_latency [body_ops]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_body_ops           (16, 64, 256, 1024 or 4096; default 256)
 *   BENCH_PARAM_total_ops          (default 16384; positive multiple of body_ops)
 *   BENCH_PARAM_prime_launches     (0 or 1; default 0)
 *   BENCH_PARAM_num_blocks         (default 1)
 *   BENCH_PARAM_threads_per_block  (default 64)
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before the primer/timed launch)
 *
 * Output: one {"type":"kernel",...} JSON line for the single timed launch,
 * carrying the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms equals it. The primer has no timing record. Fetch cost is
 * derived downstream from pairs of runs.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "../../../common_cuda/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BODY_OPS          = 256;    // FMAs in the straight-line loop body
static const int DEFAULT_TOTAL_OPS         = 16384;  // FMAs executed per thread, any body size
static const int DEFAULT_PRIME_LAUNCHES    = 0;      // cold entry
static const int DEFAULT_NUM_BLOCKS        = 1;      // one block: single SM/CU
static const int DEFAULT_THREADS_PER_BLOCK = 64;     // one AMD wavefront
static const int MAX_THREADS_PER_BLOCK     = 1024;   // launch limit on supported targets

/* Chain coefficients. They reach the kernels as runtime arguments so the
 * compiler cannot fold the chain; values keep x finite and changing. */
static const float FETCH_MUL = 1.0000001f;
static const float FETCH_ADD = 0.0000001f;

/* Relative tolerance against the host chain: covers contraction/rounding of
 * the per-thread seed, which the dependent FMA chain then carries along. */
static const double VERIFY_TOLERANCE = 5e-6;

// ---------------------------------------------------------------------------
// Code-fetch kernels
// ---------------------------------------------------------------------------

/* Straight-line bodies of exactly N dependent FMAs. */
#define FETCH_OPS_1    x = fmaf(x, a, b);
#define FETCH_OPS_4    FETCH_OPS_1 FETCH_OPS_1 FETCH_OPS_1 FETCH_OPS_1
#define FETCH_OPS_16   FETCH_OPS_4 FETCH_OPS_4 FETCH_OPS_4 FETCH_OPS_4
#define FETCH_OPS_64   FETCH_OPS_16 FETCH_OPS_16 FETCH_OPS_16 FETCH_OPS_16
#define FETCH_OPS_256  FETCH_OPS_64 FETCH_OPS_64 FETCH_OPS_64 FETCH_OPS_64
#define FETCH_OPS_1024 FETCH_OPS_256 FETCH_OPS_256 FETCH_OPS_256 FETCH_OPS_256
#define FETCH_OPS_4096 FETCH_OPS_1024 FETCH_OPS_1024 FETCH_OPS_1024 FETCH_OPS_1024

/* Per-thread chain seed, and the single store that makes the chain observable. */
#define FETCH_SEED() (1.0f + static_cast<float>(threadIdx.x) * 0.001f)
#define FETCH_STORE(value) \
    out[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = (value)

/**
 * fetch_kernel_<body_ops> — each thread runs reps * body_ops dependent FMAs.
 *
 * The loop body is body_ops straight-line FMAs; `#pragma unroll 1` keeps the
 * compiler from unrolling the loop, which would change the code footprint.
 * Check the compiled code before using a result (see README).
 *
 * @param out   Device array with one float per launched thread.
 * @param a, b  Chain coefficients (runtime values).
 * @param reps  Loop iterations; total FMAs per thread = reps * body_ops.
 */
__global__ void fetch_kernel_16(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_16 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_64(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_64 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_256(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_256 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_1024(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_1024 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_4096(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_4096 }
    FETCH_STORE(x);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/** Launch the kernel whose loop body holds body_ops FMAs. */
static void launch_fetch_kernel(int body_ops, int num_blocks,
                                int threads_per_block, float *out, int reps)
{
    const dim3 grid(num_blocks), block(threads_per_block);
    switch (body_ops) {
    case 16:   fetch_kernel_16<<<grid, block>>>(out, FETCH_MUL, FETCH_ADD, reps);   break;
    case 64:   fetch_kernel_64<<<grid, block>>>(out, FETCH_MUL, FETCH_ADD, reps);   break;
    case 256:  fetch_kernel_256<<<grid, block>>>(out, FETCH_MUL, FETCH_ADD, reps);  break;
    case 1024: fetch_kernel_1024<<<grid, block>>>(out, FETCH_MUL, FETCH_ADD, reps); break;
    case 4096: fetch_kernel_4096<<<grid, block>>>(out, FETCH_MUL, FETCH_ADD, reps); break;
    }
}

/** Host chain for one thread of a block: the value its kernel must store. */
static float reference_chain(int thread_in_block, int total_ops)
{
    float x = 1.0f + static_cast<float>(thread_in_block) * 0.001f;
    for (int i = 0; i < total_ops; ++i) x = fmaf(x, FETCH_MUL, FETCH_ADD);
    return x;
}

/**
 * Compare one launch's device output with the host chain; returns the number
 * of threads that differ. `expected` holds one value per thread of a block.
 */
static size_t count_mismatches(const float *d_out, size_t total_threads,
                               const std::vector<float> &expected,
                               const char *launch)
{
    std::vector<float> h_out(total_threads);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_threads * sizeof(float),
                        cudaMemcpyDeviceToHost));
    size_t errors = 0;
    for (size_t i = 0; i < total_threads; ++i) {
        const double want = expected[i % expected.size()];
        const double got = h_out[i];
        if (!std::isfinite(got) || !std::isfinite(want) ||
            std::fabs(got - want) > VERIFY_TOLERANCE * std::fabs(want)) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch in %s launch at thread %zu: GPU=%.9g CPU=%.9g\n",
                        launch, i, got, want);
            }
            ++errors;
        }
    }
    return errors;
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
    int body_ops = DEFAULT_BODY_OPS;
    env_val = getenv("BENCH_PARAM_body_ops");
    if (env_val) {
        body_ops = atoi(env_val);
    } else if (argc >= 2) {
        body_ops = atoi(argv[1]);
    }
    /* One kernel exists per supported body size. */
    if (body_ops != 16 && body_ops != 64 && body_ops != 256 &&
        body_ops != 1024 && body_ops != 4096) {
        fprintf(stderr, "Invalid body_ops: need 16, 64, 256, 1024 or 4096\n");
        return EXIT_FAILURE;
    }

    int total_ops = DEFAULT_TOTAL_OPS;
    env_val = getenv("BENCH_PARAM_total_ops");
    if (env_val) total_ops = atoi(env_val);
    if (total_ops <= 0 || total_ops % body_ops != 0) {
        fprintf(stderr, "Invalid total_ops: need a positive multiple of body_ops\n");
        return EXIT_FAILURE;
    }
    const int reps = total_ops / body_ops;

    int prime_launches = DEFAULT_PRIME_LAUNCHES;
    env_val = getenv("BENCH_PARAM_prime_launches");
    if (env_val) prime_launches = atoi(env_val);
    if (prime_launches != 0 && prime_launches != 1) {
        fprintf(stderr, "Invalid prime_launches: need 0 or 1\n");
        return EXIT_FAILURE;
    }

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    // Validate before multiplying: the output owns one element per thread.
    if (num_blocks <= 0 || threads_per_block <= 0 ||
        threads_per_block > MAX_THREADS_PER_BLOCK ||
        static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(float) /
                static_cast<size_t>(threads_per_block)) {
        fprintf(stderr, "Invalid or overflowing launch dimensions\n");
        return EXIT_FAILURE;
    }
    const size_t total_threads =
        static_cast<size_t>(num_blocks) * static_cast<size_t>(threads_per_block);

    /* Print device information to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    char kernel_name[32];
    snprintf(kernel_name, sizeof(kernel_name), "fetch_kernel_%d", body_ops);

    fprintf(stderr, "=== code_fetch_latency microbenchmark ===\n");
    fprintf(stderr, "Device         : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Kernel         : %s\n", kernel_name);
    fprintf(stderr, "Chain          : %d FMAs/thread = %d reps x %d body FMAs\n",
            total_ops, reps, body_ops);
    fprintf(stderr, "Grid           : %d blocks x %d threads = %zu total threads\n",
            num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "Entry policy   : optional warmer -> %d untimed primer launch(es) -> timed launch\n\n",
            prime_launches);

    /* The timed launch and the primer each own an output buffer, so a primer
     * never touches the memory the timed launch stores to. Both are set up in
     * either entry, keeping everything before the warmer identical. Every
     * element starts as NaN (all-ones bytes), so a thread that never stores
     * is caught by verification. */
    const size_t out_bytes = total_threads * sizeof(float);
    float *d_out = nullptr;
    float *d_primer_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_primer_out, out_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, out_bytes));
    CUDA_CHECK(cudaMemset(d_primer_out, 0xFF, out_bytes));

    /* Condition clocks/cache first. Nothing but the optional primer may run
     * between here and the timed launch. */
    bench_pass_warm();

    /* Resident-code entry exception: fetch this kernel's code once, untimed,
     * storing to the primer's own buffer. Retained when the clock warmer is
     * disabled (including trace collection). */
    for (int i = 0; i < prime_launches; ++i) {
        launch_fetch_kernel(body_ops, num_blocks, threads_per_block, d_primer_out, reps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    char params[256];
    snprintf(params, sizeof(params),
             "\"body_ops\":%d,\"total_ops\":%d,\"reps\":%d,"
             "\"prime_launches\":%d,\"num_blocks\":%d,\"threads_per_block\":%d,"
             "\"code_entry\":\"%s\"",
             body_ops, total_ops, reps, prime_launches, num_blocks,
             threads_per_block, prime_launches ? "primed" : "cold");

    /* Use the event timer directly so bench_run_once cannot insert another
     * warmer between the primer and the measured launch. */
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t lap = bench_time_op(&timer, [&]() {
        launch_fetch_kernel(body_ops, num_blocks, threads_per_block, d_out, reps);
    });
    bench_timer_destroy(&timer);
    bench_emit_iter(kernel_name, 0, &lap, params);
    double total_gpu_ms = lap.gpu_ms;
    fprintf(stderr, "%s: gpu = %.3f ms, host = %.3f ms\n",
            kernel_name, lap.gpu_ms, lap.host_ms);

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    /* Verify every thread's stored chain value against the host chain: the
     * timed launch on its own, and the primer when one ran. */
    std::vector<float> expected(threads_per_block);
    for (int t = 0; t < threads_per_block; ++t) {
        expected[t] = reference_chain(t, total_ops);
    }
    size_t errors = count_mismatches(d_out, total_threads, expected, "timed");
    if (prime_launches > 0) {
        errors += count_mismatches(d_primer_out, total_threads, expected, "primer");
    }
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_primer_out));
    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu stored values differ from the host chain\n", errors);
        return EXIT_FAILURE;
    }
    fprintf(stderr, "PASS: %zu threads match the host chain%s\n", total_threads,
            prime_launches > 0 ? " in the timed and primer launches" : "");

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
