/**
 * shared_mem_bandwidth.hip
 *
 * HIP port of the shared memory BANDWIDTH microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures GPU shared-memory (LDS / SMEM) read bandwidth. Each thread issues
 * UNROLL independent LDS reads per iteration into UNROLL independent register
 * accumulators, so many accesses are in flight at once and the LDS unit can be
 * saturated -- the throughput test (the predecessor, a single-accumulator
 * read->add->write recurrence, actually measured latency; it now lives in
 * shared_mem_latency).
 *
 * Forcing real LDS read traffic needs two things, because constant, loop-
 * invariant LDS data gets optimized away:
 *   1. Stage the buffer from a GLOBAL input (d_in) whose contents are opaque to
 *      the compiler, so the reads cannot be constant-folded.
 *   2. Make the read index depend on the loop counter, so the reads are not
 *      loop-invariant and cannot be hoisted.
 * d_in is filled with 1.0 on the host, so each block's sink value is exactly
 * UNROLL*inner_iters (deterministic), but the compiler must still emit
 * inner_iters*UNROLL real LDS reads.
 *
 * Two access patterns:
 *   - no_conflict : consecutive lanes -> consecutive banks (conflict-free)
 *   - conflict    : 32 lanes -> the same bank (32-way bank conflict)
 *
 * For a meaningful bandwidth number launch a full block (>=256 threads) and
 * many blocks (>= number of CUs); see params.json.
 *
 * Each pattern is timed with HIP events. 5 warmup + 1 timed launch.
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

#if defined(__NVCC__) && !__has_include(<hip/hip_runtime.h>)
#include <cuda_runtime.h>
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemset               cudaMemset
#define hipMemsetAsync          cudaMemsetAsync
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyToSymbol       cudaMemcpyToSymbol
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetLastError         cudaGetLastError
#define hipGetErrorString       cudaGetErrorString
#define hipEvent_t              cudaEvent_t
#define hipEventCreate          cudaEventCreate
#define hipEventDestroy         cudaEventDestroy
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipStream_t             cudaStream_t
#define hipStreamCreate         cudaStreamCreate
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize
#define hipDeviceProp_t         cudaDeviceProp
#define hipError_t              cudaError_t
#define hipSuccess              cudaSuccess
#define hipHostMalloc           cudaMallocHost
#define hipHostFree             cudaFreeHost
#define hipHostMallocDefault    cudaHostAllocDefault
#define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#else
#include <hip/hip_runtime.h>
#endif

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
static const int DEFAULT_NUM_BLOCKS   = 304;          // saturate the GPU (>= #CUs)
static const int DEFAULT_INNER_ITERS  = 1000;
static const int DEFAULT_WARMUP_ITERS = 5;
static const int UNROLL               = 8;            // independent accumulators
static const int SMEM_FLOATS          = 8192;         // 32 KB / 4 bytes; >= max block_size*UNROLL
static const double BYTES_PER_GB      = 1e9;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * No-conflict bandwidth kernel: UNROLL independent reads/iteration, consecutive
 * lanes -> consecutive banks. it-varying index prevents hoisting; opaque d_in
 * prevents constant-folding.
 */
__global__ void smem_bw_no_conflict(const float *d_in, float *d_sink,
                                    int inner_iters)
{
    __shared__ float smem[SMEM_FLOATS];
    const int tid = threadIdx.x;
    const int n   = blockDim.x;

    for (int i = tid; i < SMEM_FLOATS; i += n)
        smem[i] = d_in[i];
    __syncthreads();

    float acc[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        acc[u] = 0.0f;

    for (int it = 0; it < inner_iters; ++it) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = (tid + (u + it) * n) % SMEM_FLOATS;
            acc[u] += smem[idx];
        }
    }
    __syncthreads();

    float sum = 0.0f;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        sum += acc[u];

    if (tid == 0) d_sink[blockIdx.x] = sum;   // == UNROLL*inner_iters (d_in all 1.0)
}

/**
 * Conflict bandwidth kernel: all 32 lanes of a warp hit the same bank.
 */
__global__ void smem_bw_conflict(const float *d_in, float *d_sink,
                                 int inner_iters)
{
    __shared__ float smem[SMEM_FLOATS];
    const int tid  = threadIdx.x;
    const int n    = blockDim.x;
    const int lane = tid & 31;

    for (int i = tid; i < SMEM_FLOATS; i += n)
        smem[i] = d_in[i];
    __syncthreads();

    float acc[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        acc[u] = 0.0f;

    for (int it = 0; it < inner_iters; ++it) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            // stride 32 floats between lanes -> same bank (32-way conflict);
            // (u+it) shifts the bank and makes the access it-varying.
            int idx = (lane * 32 + ((u + it) & 31)) % SMEM_FLOATS;
            acc[u] += smem[idx];
        }
    }
    __syncthreads();

    float sum = 0.0f;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        sum += acc[u];

    if (tid == 0) d_sink[blockIdx.x] = sum;
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFn)(const float *, float *, int);

/**
 * Run a kernel, time it with HIP events, return time in ms and bandwidth (GB/s).
 */
static double run_benchmark(KernelFn kernel, const float *d_in, float *d_sink,
                            int block_size, int num_blocks, int inner_iters,
                            int num_warmup, double *out_bw_GBs)
{
    for (int i = 0; i < num_warmup; ++i) {
        hipLaunchKernelGGL(kernel, dim3(num_blocks), dim3(block_size),
                           0, 0, d_in, d_sink, inner_iters);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, 0));
    hipLaunchKernelGGL(kernel, dim3(num_blocks), dim3(block_size),
                       0, 0, d_in, d_sink, inner_iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    double avg_ms = (double)total_ms;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Read-only: each of block_size threads does UNROLL reads per inner-iter,
    // over num_blocks blocks. 4 bytes per read.
    double bytes = (double)UNROLL * inner_iters * block_size * num_blocks
                   * sizeof(float);
    double bw_GBs = bytes / (avg_ms * 1e-3) / BYTES_PER_GB;

    if (out_bw_GBs) *out_bw_GBs = bw_GBs;
    return avg_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

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

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device        : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Shared mem/blk: %zu bytes\n", (size_t)prop.sharedMemPerBlock);
    fprintf(stderr, "Block size    : %d threads\n", block_size);
    fprintf(stderr, "Grid size     : %d blocks\n", num_blocks);
    fprintf(stderr, "Inner iters   : %d\n", inner_iters);
    fprintf(stderr, "Access pattern: %s\n", access_pattern);
    fprintf(stderr, "Warmup iters  : %d\n\n", num_warmup);

    // Opaque source data: all 1.0, so the result is deterministic but the
    // compiler can't fold the LDS reads.
    float *h_in = (float *)malloc(SMEM_FLOATS * sizeof(float));
    for (int i = 0; i < SMEM_FLOATS; ++i) h_in[i] = 1.0f;

    float *d_in = nullptr;
    HIP_CHECK(hipMalloc(&d_in, SMEM_FLOATS * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, SMEM_FLOATS * sizeof(float),
                        hipMemcpyHostToDevice));

    float *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, num_blocks * sizeof(float)));

    double total_time_ms = 0.0;
    double max_bw = 0.0;

    bool run_no_conflict = (strcmp(access_pattern, "no_conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);
    bool run_conflict    = (strcmp(access_pattern, "conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);

    if (run_no_conflict) {
        double nc_bw = 0.0;
        double nc_ms = run_benchmark(smem_bw_no_conflict, d_in, d_sink,
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

    if (run_conflict) {
        double c_bw = 0.0;
        double c_ms = run_benchmark(smem_bw_conflict, d_in, d_sink,
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

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"bandwidth_gbps\",\"value\":%.2f}]}\n",
           total_time_ms, max_bw);

    HIP_CHECK(hipFree(d_sink));
    HIP_CHECK(hipFree(d_in));
    free(h_in);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
