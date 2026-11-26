/**
 * code_fetch_latency.cu
 *
 * Native CUDA version of the code_fetch_latency microbenchmark (rename-only port of ../hip/).
 * Compiles with nvcc for NVIDIA GPUs.
 *
 * Microbenchmark for what a kernel's code costs as a function of its size:
 * the first pass over code that has not been executed yet, and repeat passes
 * over code that has.
 *
 * Methodology:
 *   Each process launches one kernel once. Every thread runs a straight-line
 *   body of body_ops dependent operations on register values, `loops` times,
 *   in a loop that is not unrolled, and stores the result once. One kernel
 *   exists per body size and operation kind, so a launch fetches only the
 *   code of the size under test.
 *
 *   `op` selects the body operation:
 *     fma      x = fmaf(x, a, b) on float: one instruction per operation.
 *     xor_add  x = (x ^ a) + b on unsigned: two cheap instructions per
 *              operation, so the body consumes code faster than the FMA body.
 *
 *   Entry: query the selected kernel's attributes, which requests that the
 *   runtime load it without running it, then the optional vector-add
 *   clock/cache warmer, then the single timed launch. The timed launch is the
 *   first dispatch of that kernel in the process.
 *
 *   With T(N, R) the time for body_ops N and loops R, the fixed dispatch cost,
 *   the repeat-pass cost per operation and the first-pass extra cost are
 *   derived downstream from several runs (see README).
 *
 * Usage:
 *   ./code_fetch_latency [body_ops]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_body_ops           (a power of two from 16 to 32768;
 *                                   default 256)
 *   BENCH_PARAM_loops              (default 1; at least 1, with
 *                                   body_ops * loops at most 1048576)
 *   BENCH_PARAM_op                 (fma or xor_add; default fma)
 *   BENCH_PARAM_num_blocks         (default 1)
 *   BENCH_PARAM_threads_per_block  (default 64)
 *   BENCH_PARAM_vector_add_warmup  (default true; clock-boost + LLC-evict
 *                                   warmer before the timed launch)
 * Integer parameters must be whole decimal numbers in range; anything else
 * is rejected before device access.
 *
 * Output: one {"type":"kernel",...} JSON line for the single timed launch,
 * carrying the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms equals it.
 */

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const int DEFAULT_BODY_OPS          = 256;    // operations in the straight-line loop body
static const int MIN_BODY_OPS              = 16;     // smallest compiled body
static const int MAX_BODY_OPS              = 32768;  // largest compiled body
static const int DEFAULT_LOOPS             = 1;      // one pass: every operation runs once, cold
static const int MAX_CHAIN_OPS             = 1 << 20; // body_ops * loops; keeps the FMA chain finite and the host reference cheap
static const int DEFAULT_NUM_BLOCKS        = 1;      // one block: single SM/CU
static const int DEFAULT_THREADS_PER_BLOCK = 64;     // one AMD wavefront
static const int MAX_THREADS_PER_BLOCK     = 1024;   // launch limit on supported targets

/* Chain coefficients. They reach the kernels as runtime arguments so the
 * compiler cannot fold a chain. Every FMA strictly increases x, and x stays
 * finite for any admitted chain (at most MAX_CHAIN_OPS operations). */
static const float FETCH_MUL = 1.0000001f;
static const float FETCH_ADD = 0.0000001f;
/* Integer chain: x -> (x ^ FETCH_XOR) + FETCH_INC, exact with uint32 wraparound. */
static const unsigned FETCH_XOR = 0x9E3779B9u;
static const unsigned FETCH_INC = 0x7F4A7C15u;

// ---------------------------------------------------------------------------
// Code-fetch kernels
// ---------------------------------------------------------------------------

/* One body operation of each kind. */
#define FETCH_FMA      x = fmaf(x, a, b);
#define FETCH_XOR_ADD  x = (x ^ a) + b;

/* Straight-line bodies of exactly N copies of one operation, by doubling. */
#define FETCH_OPS_1(OP)     OP
#define FETCH_OPS_2(OP)     FETCH_OPS_1(OP) FETCH_OPS_1(OP)
#define FETCH_OPS_4(OP)     FETCH_OPS_2(OP) FETCH_OPS_2(OP)
#define FETCH_OPS_8(OP)     FETCH_OPS_4(OP) FETCH_OPS_4(OP)
#define FETCH_OPS_16(OP)    FETCH_OPS_8(OP) FETCH_OPS_8(OP)
#define FETCH_OPS_32(OP)    FETCH_OPS_16(OP) FETCH_OPS_16(OP)
#define FETCH_OPS_64(OP)    FETCH_OPS_32(OP) FETCH_OPS_32(OP)
#define FETCH_OPS_128(OP)   FETCH_OPS_64(OP) FETCH_OPS_64(OP)
#define FETCH_OPS_256(OP)   FETCH_OPS_128(OP) FETCH_OPS_128(OP)
#define FETCH_OPS_512(OP)   FETCH_OPS_256(OP) FETCH_OPS_256(OP)
#define FETCH_OPS_1024(OP)  FETCH_OPS_512(OP) FETCH_OPS_512(OP)
#define FETCH_OPS_2048(OP)  FETCH_OPS_1024(OP) FETCH_OPS_1024(OP)
#define FETCH_OPS_4096(OP)  FETCH_OPS_2048(OP) FETCH_OPS_2048(OP)
#define FETCH_OPS_8192(OP)  FETCH_OPS_4096(OP) FETCH_OPS_4096(OP)
#define FETCH_OPS_16384(OP) FETCH_OPS_8192(OP) FETCH_OPS_8192(OP)
#define FETCH_OPS_32768(OP) FETCH_OPS_16384(OP) FETCH_OPS_16384(OP)

/* Every compiled body size: each power of two from MIN_BODY_OPS to
 * MAX_BODY_OPS. Kernels and their dispatch are generated from this one list. */
#define FETCH_FOR_EACH_SIZE(X) \
    X(16) X(32) X(64) X(128) X(256) X(512) X(1024) X(2048) X(4096) X(8192) \
    X(16384) X(32768)

/* FMA chain seed of a block's thread: 1 + thread/1024, exact in float for
 * every launchable thread index, so device and host start from identical bits
 * whether or not a compiler contracts the multiply-add. */
#define FETCH_SEED(thread) (1.0f + static_cast<float>(thread) * 0.0009765625f)
/* Integer chain seed of a block's thread. */
#define FETCH_INT_SEED(thread) (static_cast<unsigned>(thread) * 2654435761u + 1u)
/* The single store that makes a chain observable. */
#define FETCH_STORE(value) \
    out[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = (value)

/**
 * fetch_kernel_<body_ops> — each thread runs reps * body_ops dependent FMAs.
 * fetch_xor_add_kernel_<body_ops> — the same with the integer operation.
 *
 * The loop body is body_ops straight-line operations; `unroll 1` keeps the
 * compiler from unrolling the loop, which would change the code footprint.
 * Check the compiled code before using a result (see README).
 *
 * @param out   Device array with one element per launched thread.
 * @param a, b  Chain coefficients (runtime values).
 * @param reps  Loop iterations (`loops`); operations per thread = reps * body_ops.
 */
#define FETCH_DEFINE_FMA_KERNEL(N)                                             \
    __global__ void fetch_kernel_##N(float *out, float a, float b, int reps)   \
    {                                                                          \
        float x = FETCH_SEED(threadIdx.x);                                     \
        _Pragma("unroll 1")                                                    \
        for (int r = 0; r < reps; ++r) { FETCH_OPS_##N(FETCH_FMA) }            \
        FETCH_STORE(x);                                                        \
    }

#define FETCH_DEFINE_XOR_ADD_KERNEL(N)                                         \
    __global__ void fetch_xor_add_kernel_##N(unsigned *out, unsigned a,        \
                                             unsigned b, int reps)             \
    {                                                                          \
        unsigned x = FETCH_INT_SEED(threadIdx.x);                              \
        _Pragma("unroll 1")                                                    \
        for (int r = 0; r < reps; ++r) { FETCH_OPS_##N(FETCH_XOR_ADD) }        \
        FETCH_STORE(x);                                                        \
    }

FETCH_FOR_EACH_SIZE(FETCH_DEFINE_FMA_KERNEL)
FETCH_FOR_EACH_SIZE(FETCH_DEFINE_XOR_ADD_KERNEL)

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/**
 * Request that the runtime load the selected kernel, without dispatching it,
 * by querying its attributes. Runtimes may defer loading a kernel's code to
 * its first launch; the query is made before timing so that work need not
 * fall inside the timed bracket. Whether it fully pre-loads a kernel is
 * runtime-specific (see README).
 */
static void load_fetch_kernel(bool xor_add, int body_ops)
{
    const void *kernel = nullptr;
    switch (body_ops) {
#define FETCH_CASE(N)                                                          \
    case N:                                                                    \
        kernel = xor_add ? reinterpret_cast<const void *>(fetch_xor_add_kernel_##N) \
                         : reinterpret_cast<const void *>(fetch_kernel_##N);   \
        break;
    FETCH_FOR_EACH_SIZE(FETCH_CASE)
#undef FETCH_CASE
    }
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
}

/** Launch the selected kernel; `out` holds one 4-byte element per thread. */
static void launch_fetch_kernel(bool xor_add, int body_ops, int num_blocks,
                                int threads_per_block, void *out, int reps)
{
    const dim3 grid(num_blocks), block(threads_per_block);
    switch (body_ops) {
#define FETCH_CASE(N)                                                          \
    case N:                                                                    \
        if (xor_add) {                                                         \
            fetch_xor_add_kernel_##N<<<grid, block>>>(                         \
                static_cast<unsigned *>(out), FETCH_XOR, FETCH_INC, reps);     \
        } else {                                                               \
            fetch_kernel_##N<<<grid, block>>>(                                 \
                static_cast<float *>(out), FETCH_MUL, FETCH_ADD, reps);        \
        }                                                                      \
        break;
    FETCH_FOR_EACH_SIZE(FETCH_CASE)
#undef FETCH_CASE
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

/** Host FMA chain for one thread of a block: the value its kernel must store. */
static float reference_chain(int thread_in_block, int chain_ops)
{
    float x = FETCH_SEED(thread_in_block);
    for (int i = 0; i < chain_ops; ++i) x = fmaf(x, FETCH_MUL, FETCH_ADD);
    return x;
}

/** Host integer chain for one thread of a block. */
static unsigned reference_int_chain(int thread_in_block, int chain_ops)
{
    unsigned x = FETCH_INT_SEED(thread_in_block);
    for (int i = 0; i < chain_ops; ++i) x = (x ^ FETCH_XOR) + FETCH_INC;
    return x;
}

/**
 * A stored FMA value is right only when it is finite and exactly the host
 * chain's value. Seed and FMAs are exact on both sides and every FMA changes
 * x, so a chain with a missing or extra loop iteration never matches.
 */
static bool chain_value_matches(float got, float want)
{
    return std::isfinite(got) && got == want;
}

/**
 * Compare the launch's device output with the host chain; returns the number
 * of threads that differ.
 */
static size_t count_mismatches(bool xor_add, const void *d_out, size_t total_threads,
                               int threads_per_block, int chain_ops)
{
    std::vector<unsigned> h_bits(total_threads);
    CUDA_CHECK(cudaMemcpy(h_bits.data(), d_out, total_threads * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
    size_t errors = 0;
    std::vector<float> want_fma(xor_add ? 0 : threads_per_block);
    std::vector<unsigned> want_int(xor_add ? threads_per_block : 0);
    for (int t = 0; t < threads_per_block; ++t) {
        if (xor_add) want_int[t] = reference_int_chain(t, chain_ops);
        else want_fma[t] = reference_chain(t, chain_ops);
    }
    for (size_t i = 0; i < total_threads; ++i) {
        const size_t t = i % static_cast<size_t>(threads_per_block);
        bool match;
        float got_fma = 0.0f;
        if (xor_add) {
            match = h_bits[i] == want_int[t];
        } else {
            memcpy(&got_fma, &h_bits[i], sizeof(got_fma));
            match = chain_value_matches(got_fma, want_fma[t]);
        }
        if (!match) {
            if (errors < 10 && xor_add) {
                fprintf(stderr, "Mismatch at thread %zu: GPU=0x%08x CPU=0x%08x\n",
                        i, h_bits[i], want_int[t]);
            } else if (errors < 10) {
                fprintf(stderr, "Mismatch at thread %zu: GPU=%.9g CPU=%.9g\n",
                        i, got_fma, want_fma[t]);
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
    /* One kernel exists per supported body size: every power of two from
     * MIN_BODY_OPS to MAX_BODY_OPS. */
    if ((env_val && !parse_int(env_val, MIN_BODY_OPS, MAX_BODY_OPS, &body_ops)) ||
        (body_ops & (body_ops - 1)) != 0) {
        fprintf(stderr, "Invalid body_ops: need a power of two from %d to %d\n",
                MIN_BODY_OPS, MAX_BODY_OPS);
        return EXIT_FAILURE;
    }

    int loops = DEFAULT_LOOPS;
    env_val = getenv("BENCH_PARAM_loops");
    if (env_val && !parse_int(env_val, 1, MAX_CHAIN_OPS / body_ops, &loops)) {
        fprintf(stderr, "Invalid loops: need a positive integer with body_ops * loops at most %d\n",
                MAX_CHAIN_OPS);
        return EXIT_FAILURE;
    }
    const int chain_ops = body_ops * loops;

    const char *op = "fma";
    env_val = getenv("BENCH_PARAM_op");
    if (env_val) op = env_val;
    const bool xor_add = strcmp(op, "xor_add") == 0;
    if (!xor_add && strcmp(op, "fma") != 0) {
        fprintf(stderr, "Invalid op: need fma or xor_add\n");
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

    // Validate before multiplying: the output buffer owns one 4-byte element per thread.
    if (static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(unsigned) /
                static_cast<size_t>(threads_per_block)) {
        fprintf(stderr, "Launch dimensions overflow the output size\n");
        return EXIT_FAILURE;
    }
    const size_t total_threads =
        static_cast<size_t>(num_blocks) * static_cast<size_t>(threads_per_block);

    /* Print device information to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    char kernel_name[40];
    snprintf(kernel_name, sizeof(kernel_name), "%s_%d",
             xor_add ? "fetch_xor_add_kernel" : "fetch_kernel", body_ops);

    fprintf(stderr, "=== code_fetch_latency microbenchmark ===\n");
    fprintf(stderr, "Device         : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Kernel         : %s\n", kernel_name);
    fprintf(stderr, "Chain          : %d %s ops/thread = %d loops x %d body ops\n",
            chain_ops, op, loops, body_ops);
    fprintf(stderr, "Grid           : %d blocks x %d threads = %zu total threads\n",
            num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "Entry policy   : kernel attribute query -> optional warmer -> timed launch\n\n");

    /* Query the selected kernel's attributes before timing to request that
     * the runtime load it without running it. Whether that fully pre-loads
     * the kernel is runtime-specific. The timed launch is still the kernel's
     * first dispatch. */
    load_fetch_kernel(xor_add, body_ops);

    /* One 4-byte element per thread, float or unsigned by operation kind.
     * Every byte starts as 0xFF: a float thread that never stores leaves a
     * NaN, which verification rejects. */
    const size_t out_bytes = total_threads * sizeof(unsigned);
    void *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, out_bytes));

    char params[256];
    snprintf(params, sizeof(params),
             "\"body_ops\":%d,\"loops\":%d,\"op\":\"%s\","
             "\"num_blocks\":%d,\"threads_per_block\":%d",
             body_ops, loops, op, num_blocks, threads_per_block);

    /* Optional warmer, then the single timed launch: the kernel's first
     * dispatch in this process. */
    const double total_gpu_ms = bench_run_once(kernel_name, params, [&]() {
        launch_fetch_kernel(xor_add, body_ops, num_blocks, threads_per_block, d_out, loops);
    });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    /* Verify every thread's stored chain value against the host chain. */
    const size_t errors =
        count_mismatches(xor_add, d_out, total_threads, threads_per_block, chain_ops);
    CUDA_CHECK(cudaFree(d_out));
    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu stored values differ from the host chain\n", errors);
        return EXIT_FAILURE;
    }
    fprintf(stderr, "PASS: %zu threads match the host chain\n", total_threads);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
