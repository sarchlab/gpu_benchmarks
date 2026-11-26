/**
 * code_fetch_latency.hip
 *
 * HIP version of the code_fetch_latency microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for the first-dispatch cost of a kernel as a function of its
 * code size; built to expose the cost of fetching code that has not been
 * executed yet.
 *
 * Methodology:
 *   Every thread runs one dependent chain of total_ops fused multiply-adds,
 *   x = fmaf(x, a, b), on register values. The chain is laid out as a
 *   straight-line body of body_ops FMAs inside a loop of
 *   reps = total_ops / body_ops iterations. The number of dependent FMAs
 *   executed is therefore the same for every body_ops while the static code
 *   footprint changes; loop-control work is not the same (smaller bodies run
 *   more iterations), so cold and primed times are paired within one body
 *   size. One kernel exists per supported body size, so a launch fetches only
 *   the code of the size under test.
 *
 *   Entry policy: query the selected kernel's attributes, which requests
 *   that the runtime load it without running it, then the optional vector-add
 *   clock/cache warmer, then prime_launches untimed launches of the same
 *   kernel, then one timed launch. Whether the query fully pre-loads the
 *   kernel is established by target qualification. With
 *   prime_launches=0 the timed launch is the first dispatch of that kernel in
 *   the process (the suite's default cold entry). With prime_launches=1 the
 *   kernel was dispatched once already: an explicit primed exception to cold
 *   entry. Keep the primer during tracing. A primer runs the same kernel
 *   with the same reps but stores to its own buffer, so the timed launch does
 *   not reuse a primer's output. The raw result is first-dispatch minus
 *   primed-dispatch time at one body size; reading its growth with code size
 *   as instruction-fetch cost requires target qualification (see README).
 *
 * Usage:
 *   ./code_fetch_latency [body_ops]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_body_ops           (16, 64, 256, 1024 or 4096; default 256)
 *   BENCH_PARAM_total_ops          (default 16384; positive multiple of
 *                                   body_ops, at most 1048576)
 *   BENCH_PARAM_prime_launches     (0 or 1; default 0)
 *   BENCH_PARAM_num_blocks         (default 1)
 *   BENCH_PARAM_threads_per_block  (default 64)
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before the primer/timed launch)
 * Integer parameters must be whole decimal numbers in range; anything else
 * is rejected before device access.
 *
 * Output: one {"type":"kernel",...} JSON line for the single timed launch,
 * carrying the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms equals it. The primer has no timing record. The
 * first-dispatch minus primed-dispatch difference is derived downstream from
 * pairs of runs.
 */

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "../../../common_hip/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BODY_OPS          = 256;    // FMAs in the straight-line loop body
static const int DEFAULT_TOTAL_OPS         = 16384;  // FMAs executed per thread, any body size
static const int MAX_TOTAL_OPS             = 1 << 20; // keeps the chain finite and the host reference cheap
static const int DEFAULT_PRIME_LAUNCHES    = 0;      // cold entry
static const int DEFAULT_NUM_BLOCKS        = 1;      // one block: single SM/CU
static const int DEFAULT_THREADS_PER_BLOCK = 64;     // one AMD wavefront
static const int MAX_THREADS_PER_BLOCK     = 1024;   // launch limit on supported targets

/* Chain coefficients. They reach the kernels as runtime arguments so the
 * compiler cannot fold the chain. Every FMA strictly increases x, and x stays
 * finite for any admitted total_ops (at most MAX_TOTAL_OPS). */
static const float FETCH_MUL = 1.0000001f;
static const float FETCH_ADD = 0.0000001f;

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

/* Chain seed of a block's thread: 1 + thread/1024, exact in float for every
 * launchable thread index, so device and host start from identical bits
 * whether or not a compiler contracts the multiply-add. */
#define FETCH_SEED(thread) (1.0f + static_cast<float>(thread) * 0.0009765625f)
/* The single store that makes the chain observable. */
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
    float x = FETCH_SEED(threadIdx.x);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_16 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_64(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED(threadIdx.x);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_64 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_256(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED(threadIdx.x);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_256 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_1024(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED(threadIdx.x);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_1024 }
    FETCH_STORE(x);
}

__global__ void fetch_kernel_4096(float *out, float a, float b, int reps)
{
    float x = FETCH_SEED(threadIdx.x);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) { FETCH_OPS_4096 }
    FETCH_STORE(x);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/**
 * Request that the runtime load the kernel whose loop body holds body_ops
 * FMAs, without dispatching it, by querying its attributes. Runtimes may defer
 * loading a kernel's code to its first launch; the query is made before timing
 * in both entries so that work need not fall inside the timed bracket. Whether
 * it fully pre-loads a kernel is runtime-specific (see README).
 */
static void load_fetch_kernel(int body_ops)
{
    const void *kernel = nullptr;
    switch (body_ops) {
    case 16:   kernel = reinterpret_cast<const void *>(fetch_kernel_16);   break;
    case 64:   kernel = reinterpret_cast<const void *>(fetch_kernel_64);   break;
    case 256:  kernel = reinterpret_cast<const void *>(fetch_kernel_256);  break;
    case 1024: kernel = reinterpret_cast<const void *>(fetch_kernel_1024); break;
    case 4096: kernel = reinterpret_cast<const void *>(fetch_kernel_4096); break;
    }
    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, kernel));
}

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

/**
 * Parse a whole decimal integer in [min_value, max_value]. False for empty,
 * malformed, trailing-junk or out-of-range text; *value is then unchanged.
 */
static bool parse_int(const char *text, int min_value, int max_value, int *value)
{
    if (!text || !*text) return false;
    char *end = nullptr;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno || *end || parsed < min_value || parsed > max_value) return false;
    *value = static_cast<int>(parsed);
    return true;
}

/** Host chain for one thread of a block: the value its kernel must store. */
static float reference_chain(int thread_in_block, int total_ops)
{
    float x = FETCH_SEED(thread_in_block);
    for (int i = 0; i < total_ops; ++i) x = fmaf(x, FETCH_MUL, FETCH_ADD);
    return x;
}

/**
 * A stored value is right only when it is finite and exactly the host chain's
 * value. Seed and FMAs are exact on both sides and every FMA changes x, so a
 * chain with a missing or extra loop iteration never matches.
 */
static bool chain_value_matches(float got, float want)
{
    return std::isfinite(got) && got == want;
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
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, total_threads * sizeof(float),
                        hipMemcpyDeviceToHost));
    size_t errors = 0;
    for (size_t i = 0; i < total_threads; ++i) {
        const float want = expected[i % expected.size()];
        const float got = h_out[i];
        if (!chain_value_matches(got, want)) {
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

    /* Read parameters: env vars take priority, then argv, then defaults.
     * Present values must parse completely and lie in range. */
    int body_ops = DEFAULT_BODY_OPS;
    env_val = getenv("BENCH_PARAM_body_ops");
    if (!env_val && argc >= 2) env_val = argv[1];
    /* One kernel exists per supported body size. */
    if ((env_val && !parse_int(env_val, 16, 4096, &body_ops)) ||
        (body_ops != 16 && body_ops != 64 && body_ops != 256 &&
         body_ops != 1024 && body_ops != 4096)) {
        fprintf(stderr, "Invalid body_ops: need 16, 64, 256, 1024 or 4096\n");
        return EXIT_FAILURE;
    }

    int total_ops = DEFAULT_TOTAL_OPS;
    env_val = getenv("BENCH_PARAM_total_ops");
    if ((env_val && !parse_int(env_val, 1, MAX_TOTAL_OPS, &total_ops)) ||
        total_ops % body_ops != 0) {
        fprintf(stderr, "Invalid total_ops: need a positive multiple of body_ops, at most %d\n",
                MAX_TOTAL_OPS);
        return EXIT_FAILURE;
    }
    const int reps = total_ops / body_ops;

    int prime_launches = DEFAULT_PRIME_LAUNCHES;
    env_val = getenv("BENCH_PARAM_prime_launches");
    if (env_val && !parse_int(env_val, 0, 1, &prime_launches)) {
        fprintf(stderr, "Invalid prime_launches: need 0 or 1\n");
        return EXIT_FAILURE;
    }

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val && !parse_int(env_val, 1, INT_MAX, &num_blocks)) {
        fprintf(stderr, "Invalid num_blocks: need a positive integer\n");
        return EXIT_FAILURE;
    }

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val && !parse_int(env_val, 1, MAX_THREADS_PER_BLOCK, &threads_per_block)) {
        fprintf(stderr, "Invalid threads_per_block: need 1 to %d\n", MAX_THREADS_PER_BLOCK);
        return EXIT_FAILURE;
    }

    // Validate before multiplying: each output buffer owns one element per thread.
    if (static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(float) /
                static_cast<size_t>(threads_per_block)) {
        fprintf(stderr, "Launch dimensions overflow the output size\n");
        return EXIT_FAILURE;
    }
    const size_t total_threads =
        static_cast<size_t>(num_blocks) * static_cast<size_t>(threads_per_block);

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    char kernel_name[32];
    snprintf(kernel_name, sizeof(kernel_name), "fetch_kernel_%d", body_ops);

    fprintf(stderr, "=== code_fetch_latency microbenchmark ===\n");
    fprintf(stderr, "Device         : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Kernel         : %s\n", kernel_name);
    fprintf(stderr, "Chain          : %d FMAs/thread = %d reps x %d body FMAs\n",
            total_ops, reps, body_ops);
    fprintf(stderr, "Grid           : %d blocks x %d threads = %zu total threads\n",
            num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "Entry policy   : kernel attribute query -> optional warmer -> %d untimed primer launch(es) -> timed launch\n\n",
            prime_launches);

    /* Query the selected kernel's attributes before timing, in either entry,
     * to request that the runtime load it without running it. Whether that
     * fully pre-loads the kernel is established by target qualification. The
     * cold timed launch is still the kernel's first dispatch. */
    load_fetch_kernel(body_ops);

    /* The timed launch and the primer each own an output buffer, so a primer
     * never touches the memory the timed launch stores to. Both are set up in
     * either entry, keeping everything before the warmer identical. Every
     * element starts as NaN (all-ones bytes), so a thread that never stores
     * is caught by verification. */
    const size_t out_bytes = total_threads * sizeof(float);
    float *d_out = nullptr;
    float *d_primer_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, out_bytes));
    HIP_CHECK(hipMalloc(&d_primer_out, out_bytes));
    HIP_CHECK(hipMemset(d_out, 0xFF, out_bytes));
    HIP_CHECK(hipMemset(d_primer_out, 0xFF, out_bytes));

    /* Condition clocks/cache first. Nothing but the optional primer may run
     * between here and the timed launch. */
    bench_pass_warm();

    /* Primed entry exception: dispatch this kernel once, untimed, storing to
     * the primer's own buffer. Retained when the clock warmer is disabled
     * (including trace collection). */
    for (int i = 0; i < prime_launches; ++i) {
        launch_fetch_kernel(body_ops, num_blocks, threads_per_block, d_primer_out, reps);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
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
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_primer_out));
    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu stored values differ from the host chain\n", errors);
        return EXIT_FAILURE;
    }
    fprintf(stderr, "PASS: %zu threads match the host chain%s\n", total_threads,
            prime_launches > 0 ? " in the timed and primer launches" : "");

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
