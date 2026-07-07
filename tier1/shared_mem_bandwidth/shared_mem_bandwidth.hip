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
 * Each pattern is timed with HIP events.  5 warmup + 5 timed iterations
 * are performed and the average is reported.
 *
 * Usage:
 *   ./shared_mem_bandwidth
 *
 * Environment variables (override defaults):
 *   BENCH_PARAM_inner_iters
 *   BENCH_PARAM_block_size
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_access_pattern   ("no_conflict", "conflict", or "both")
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../common/hip_cuda_compat.h"

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _err = (call);                                              \
        if (_err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BLOCK_SIZE   = 256;
static const int DEFAULT_NUM_BLOCKS   = 128;
static const int DEFAULT_INNER_ITERS  = 1000;
static const int DEFAULT_WARMUP_ITERS  = 5;
static const int SMEM_FLOATS          = 12288;         // 48 KB / 4 bytes
static const double BYTES_PER_GB      = 1e9;

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
 * Run a kernel, time it, return average time in ms and bandwidth.
 */
static double run_benchmark(KernelFn kernel, float *d_sink,
                            int block_size, int num_blocks, int inner_iters,
                            int num_warmup,
                            double *out_bw_GBs)
{
    // Warmup
    for (int i = 0; i < num_warmup; ++i) {
        kernel<<<dim3(num_blocks), dim3(block_size)>>>(d_sink, inner_iters);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, 0));
    kernel<<<dim3(num_blocks), dim3(block_size)>>>(d_sink, inner_iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    double avg_ms = (double)total_ms;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Bandwidth calculation
    double bytes_per_launch = 2.0 * SMEM_FLOATS * sizeof(float)
                              * (double)inner_iters * num_blocks;
    double bw_GBs = bytes_per_launch / (avg_ms * 1e-3) / BYTES_PER_GB;

    if (out_bw_GBs) *out_bw_GBs = bw_GBs;
    return avg_ms;
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

    const char* access_pattern = "both";
    env_val = getenv("BENCH_PARAM_access_pattern");
    if (env_val) access_pattern = env_val;

    int num_warmup = DEFAULT_WARMUP_ITERS;

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
    fprintf(stderr, "Access pattern: %s\n", access_pattern);
    fprintf(stderr, "Warmup iters  : %d\n\n", num_warmup);

    /* Allocate device sink buffer. */
    float *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, num_blocks * sizeof(float)));

    double total_time_ms = 0.0;
    double max_bw = 0.0;

    bool run_no_conflict = (strcmp(access_pattern, "no_conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);
    bool run_conflict    = (strcmp(access_pattern, "conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);

    /* --- No-conflict --- */
    if (run_no_conflict) {
        double nc_bw = 0.0;
        double nc_ms = run_benchmark(smem_bw_no_conflict, d_sink,
                                     block_size, num_blocks, inner_iters,
                                     num_warmup, &nc_bw);
        fprintf(stderr, "no_conflict    : %.3f ms  %.2f GB/s\n", nc_ms, nc_bw);
        total_time_ms += nc_ms;
        if (nc_bw > max_bw) max_bw = nc_bw;

        printf("{\"type\":\"kernel\",\"name\":\"smem_bw_no_conflict\",\"time_ms\":%.6f,"
               "\"params\":{\"inner_iters\":%d,\"block_size\":%d,"
               "\"num_blocks\":%d,\"access_pattern\":\"%s\"}}\n",
               nc_ms, inner_iters, block_size, num_blocks, access_pattern);
    }

    /* --- Conflict --- */
    if (run_conflict) {
        double c_bw = 0.0;
        double c_ms = run_benchmark(smem_bw_conflict, d_sink,
                                    block_size, num_blocks, inner_iters,
                                    num_warmup, &c_bw);
        fprintf(stderr, "conflict       : %.3f ms  %.2f GB/s\n", c_ms, c_bw);
        total_time_ms += c_ms;
        if (c_bw > max_bw) max_bw = c_bw;

        printf("{\"type\":\"kernel\",\"name\":\"smem_bw_conflict\",\"time_ms\":%.6f,"
               "\"params\":{\"inner_iters\":%d,\"block_size\":%d,"
               "\"num_blocks\":%d,\"access_pattern\":\"%s\"}}\n",
               c_ms, inner_iters, block_size, num_blocks, access_pattern);
    }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"bandwidth_gbps\",\"value\":%.2f}]}\n",
           total_time_ms, max_bw);

    HIP_CHECK(hipFree(d_sink));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
