/**
 * shared_mem_bandwidth.hip
 *
 * HIP port of the shared memory bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures GPU shared memory (LDS / SMEM) bandwidth with two access
 * patterns:
 *   - no_conflict : each thread in a warp accesses a different bank
 *                   (stride-1, sequential)
 *   - conflict    : all threads in a warp access the same bank
 *                   (stride-32 between lanes)
 *
 * Each selected pattern is launched EXACTLY ONCE and timed with HIP events;
 * there is no in-program warmup or iteration loop. When
 * BENCH_PARAM_vector_add_warmup is on (default), a vector-add warmer runs
 * immediately before the timed launch to boost the GPU clock and evict the
 * last-level cache, so the kernel is measured cold-cache at a boosted clock.
 * Repetition is the harness's job (gpubench --reps).
 *
 * Usage:
 *   ./shared_mem_bandwidth
 *
 * Environment variables (override defaults):
 *   BENCH_PARAM_inner_iters
 *   BENCH_PARAM_block_size
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_access_pattern   ("no_conflict", "conflict", or "both")
 *
 * Output: one {"type":"kernel",...} JSON line per selected pattern carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of the recorded GPU event times across every
 * pattern that ran. Derived metrics (GB/s) are intentionally not computed
 * here — derive them downstream from time and the work-descriptor params.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../common/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BLOCK_SIZE   = 256;
// Default 1 block = single-SM/CU measurement: shared memory (LDS) is a
// per-SM resource with no cross-SM interaction, so device peak ≈ per-SM
// value × SM count (an upper bound — an all-SM load throttles below
// single-SM boost clock).  Raise BENCH_PARAM_num_blocks (e.g. >= 8 × SM
// count) for a device-wide run.
static const int DEFAULT_NUM_BLOCKS   = 1;
static const int DEFAULT_INNER_ITERS  = 1000;
static const int SMEM_FLOATS          = 12288;         // 48 KB / 4 bytes

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * No-conflict kernel.
 */
__global__ void smem_bw_no_conflict(float *d_sink, int inner_iters)
{
    __shared__ float smem[SMEM_FLOATS];

    const int tid = threadIdx.x;
    const int n   = blockDim.x;

    // Initialise shared memory
    for (int i = tid; i < SMEM_FLOATS; i += n)
        smem[i] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int it = 0; it < inner_iters; ++it) {
        for (int i = tid; i < SMEM_FLOATS; i += n) {
            acc += smem[i];   // read
            smem[i] = acc;    // write
        }
    }
    __syncthreads();

    // Prevent dead-code elimination
    if (tid == 0) d_sink[blockIdx.x] = acc;
}

/**
 * Conflict kernel.
 */
__global__ void smem_bw_conflict(float *d_sink, int inner_iters)
{
    __shared__ float smem[SMEM_FLOATS];

    const int tid  = threadIdx.x;
    const int n    = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = n >> 5;              // number of warps

    // Initialise shared memory (normal stride-1)
    for (int i = tid; i < SMEM_FLOATS; i += n)
        smem[i] = 1.0f;
    __syncthreads();

    const int elems_per_thread = SMEM_FLOATS / n;   // e.g. 48
    float acc = 0.0f;

    for (int it = 0; it < inner_iters; ++it) {
        for (int j = 0; j < elems_per_thread; ++j) {
            int idx = (lane * 32 + warp + j * nw) % SMEM_FLOATS;
            acc += smem[idx];   // read  — all lanes hit same bank
            smem[idx] = acc;    // write — all lanes hit same bank
        }
    }
    __syncthreads();

    if (tid == 0) d_sink[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFn)(float *, int);

/**
 * Run a single access pattern: one timed launch (preceded by the cold-cache
 * clock warmer when enabled), emitted as one kernel record. Returns that
 * launch's GPU event time.
 */
static double run_benchmark(const char *kernel_name,
                            KernelFn kernel, float *d_sink,
                            int block_size, int num_blocks, int inner_iters,
                            const char *access_pattern)
{
    char params[192];
    snprintf(params, sizeof(params),
             "\"inner_iters\":%d,\"block_size\":%d,\"num_blocks\":%d,"
             "\"access_pattern\":\"%s\"",
             inner_iters, block_size, num_blocks, access_pattern);

    return bench_run_once(kernel_name, params, [&]() {
        kernel<<<dim3(num_blocks), dim3(block_size)>>>(d_sink, inner_iters);
    });
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters from env vars with defaults. */
    int inner_iters = DEFAULT_INNER_ITERS;
    env_val = getenv("BENCH_PARAM_inner_iters");
    if (env_val) inner_iters = atoi(env_val);

    int block_size = DEFAULT_BLOCK_SIZE;
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    const char* access_pattern = "no_conflict";
    env_val = getenv("BENCH_PARAM_access_pattern");
    if (env_val) access_pattern = env_val;

    /* Print device info to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device        : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Shared mem/blk: %zu bytes\n",
           (size_t)prop.sharedMemPerBlock);
    fprintf(stderr, "Block size    : %d threads\n", block_size);
    fprintf(stderr, "Grid size     : %d blocks\n", num_blocks);
    fprintf(stderr, "Inner iters   : %d\n", inner_iters);
    fprintf(stderr, "Access pattern: %s\n\n", access_pattern);

    /* Allocate device sink buffer. */
    float *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, num_blocks * sizeof(float)));

    bool run_no_conflict = (strcmp(access_pattern, "no_conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);
    bool run_conflict    = (strcmp(access_pattern, "conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);

    double total_gpu_ms = 0.0;

    /* --- No-conflict --- */
    if (run_no_conflict) {
        total_gpu_ms += run_benchmark("smem_bw_no_conflict",
                                      smem_bw_no_conflict, d_sink,
                                      block_size, num_blocks, inner_iters,
                                      access_pattern);
    }

    /* --- Conflict --- */
    if (run_conflict) {
        total_gpu_ms += run_benchmark("smem_bw_conflict",
                                      smem_bw_conflict, d_sink,
                                      block_size, num_blocks, inner_iters,
                                      access_pattern);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_sink));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
