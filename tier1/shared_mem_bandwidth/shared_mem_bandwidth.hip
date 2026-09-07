/**
 * shared_mem_bandwidth.hip
 *
 * HIP port of the shared memory bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures GPU shared memory (LDS / SMEM) bandwidth with two access
 * patterns:
 *   - no_conflict : consecutive words within each logical 32-lane group
 *   - conflict    : distinct words with stride 32 within that group
 * Bank aliasing depends on the target bank count and service phases; see README.
 *
 * Each selected pattern is launched EXACTLY ONCE and timed with HIP events;
 * there are no internal measurement repetitions. When
 * BENCH_PARAM_vector_add_warmup is on (default), a vector-add warmer runs
 * immediately before the timed launch to raise clocks and disturb cache state;
 * qualify its effect on the target. Shared initialization is inside the kernel.
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
#include <limits>

#include "../../common/bench_timing.h"
#include "shared_mem_pattern.h"

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BLOCK_SIZE   = 256;
// Default 1 block = single-SM/CU measurement: shared memory (LDS) is a
// per-SM resource. Multiplication by SM count is an extrapolation, not a
// measured device-wide rate or proven upper bound. A device-wide claim needs
// its own qualified grid and measurement.
static const int DEFAULT_NUM_BLOCKS   = 1;
static const int DEFAULT_INNER_ITERS  = 1000;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Both patterns execute the same dependent read/update/write workload. Only
// the mapping from a logical element to its shared-memory address differs.
template <bool Conflict>
__device__ __forceinline__ void smem_bw_pattern(float *d_sink, int inner_iters)
{
    // Volatile preserves the requested shared loads/stores in the compiled loop.
    __shared__ volatile float smem[SMEM_FLOATS];
    const int tid = threadIdx.x;
    const int n = blockDim.x;

    for (int i = tid; i < SMEM_FLOATS; i += n) smem[i] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int it = 0; it < inner_iters; ++it) {
        for (int i = tid; i < SMEM_FLOATS; i += n) {
            const int index = Conflict ? smem_conflict_index(i) : i;
            // A bounded dependent recurrence: acc and every element stay in
            // [0,1]. The former running sum overflowed on the declared sweep.
            acc = 0.5f * (acc + smem[index]);
            smem[index] = acc;
        }
    }
    __syncthreads();

    const size_t global = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    d_sink[global] = acc;
}

__global__ void smem_bw_no_conflict(float *d_sink, int inner_iters)
{
    smem_bw_pattern<false>(d_sink, inner_iters);
}

__global__ void smem_bw_conflict(float *d_sink, int inner_iters)
{
    smem_bw_pattern<true>(d_sink, inner_iters);
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFn)(float *, int);

/**
 * Run a single access pattern: one timed launch (preceded by the untimed
 * warmer when enabled), emitted as one kernel record. Returns that
 * launch's GPU event time.
 */
static double run_benchmark(const char *kernel_name,
                            KernelFn kernel, float *d_sink,
                            int block_size, int num_blocks, int inner_iters,
                            const char *access_pattern, int warp_size,
                            int lane_stride_bytes)
{
    char params[384];
    snprintf(params, sizeof(params),
             "\"inner_iters\":%d,\"block_size\":%d,\"num_blocks\":%d,"
             "\"access_pattern\":\"%s\",\"shared_memory_bytes\":%zu,"
             "\"word_bytes\":%zu,\"logical_lane_group_size\":%d,"
             "\"lane_stride_bytes\":%d,\"warp_size\":%d",
             inner_iters, block_size, num_blocks, access_pattern,
             SMEM_FLOATS * sizeof(float), sizeof(float), SMEM_LANE_GROUP,
             lane_stride_bytes, warp_size);

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

    if (inner_iters <= 0 || num_blocks <= 0 || block_size <= 0 ||
        block_size % SMEM_LANE_GROUP != 0 ||
        static_cast<size_t>(num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(float) /
                static_cast<size_t>(block_size)) {
        fprintf(stderr, "Invalid iterations or launch dimensions (block_size must be a positive multiple of 32)\n");
        return EXIT_FAILURE;
    }
    const size_t total_threads = static_cast<size_t>(num_blocks) * block_size;
    bool run_no_conflict = (strcmp(access_pattern, "no_conflict") == 0 ||
                           strcmp(access_pattern, "both") == 0);
    bool run_conflict = (strcmp(access_pattern, "conflict") == 0 ||
                         strcmp(access_pattern, "both") == 0);
    if (!run_no_conflict && !run_conflict) {
        fprintf(stderr, "Invalid access_pattern\n");
        return EXIT_FAILURE;
    }

    /* Print device info to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    if (block_size > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block_size exceeds device limit %d\n", prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device        : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Shared mem/blk: %zu bytes\n",
           (size_t)prop.sharedMemPerBlock);
    fprintf(stderr, "Block size    : %d threads\n", block_size);
    fprintf(stderr, "Grid size     : %d blocks\n", num_blocks);
    fprintf(stderr, "Inner iters   : %d\n", inner_iters);
    fprintf(stderr, "Access pattern: %s\n", access_pattern);
    fprintf(stderr, "Device warp   : %d lanes (logical address groups: %d)\n\n",
            prop.warpSize, SMEM_LANE_GROUP);

    /* Allocate device sink buffer. */
    float *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, total_threads * sizeof(float)));

    double total_gpu_ms = 0.0;

    /* --- No-conflict --- */
    if (run_no_conflict) {
        total_gpu_ms += run_benchmark("smem_bw_no_conflict",
                                      smem_bw_no_conflict, d_sink,
                                      block_size, num_blocks, inner_iters,
                                      access_pattern, prop.warpSize, sizeof(float));
    }

    /* --- Conflict --- */
    if (run_conflict) {
        total_gpu_ms += run_benchmark("smem_bw_conflict",
                                      smem_bw_conflict, d_sink,
                                      block_size, num_blocks, inner_iters,
                                      access_pattern, prop.warpSize,
                                      SMEM_LANE_GROUP * sizeof(float));
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    /* Clean up. */
    HIP_CHECK(hipFree(d_sink));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
