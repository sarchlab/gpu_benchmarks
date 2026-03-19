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
 */

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

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
// Constants
// ---------------------------------------------------------------------------

static const int BLOCK_SIZE    = 256;           // threads per block
static const int NUM_BLOCKS    = 128;           // grid size
static const int INNER_ITERS   = 1000;          // inner-loop iterations
static const int WARMUP_ITERS  = 5;             // outer warmup launches
static const int TIMED_ITERS   = 5;             // outer timed launches
static const int SMEM_FLOATS   = 12288;         // 48 KB / 4 bytes
static const double BYTES_PER_GB = 1e9;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * No-conflict kernel.
 *
 * Each thread strides through shared memory with stride = blockDim.x.
 * Within a warp, consecutive lanes access consecutive addresses so every
 * memory transaction hits 32 different banks — zero bank conflicts.
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
 *
 * Accesses are remapped so that within every warp all 32 lanes hit the
 * same bank on each memory transaction.  The total number of reads and
 * writes per thread is identical to the no-conflict kernel so the
 * bandwidth numbers are directly comparable.
 *
 * Index mapping:  idx = (lane * 32 + warp_id + j * num_warps) % SMEM_FLOATS
 *   bank = idx % 32 = (warp_id + j * num_warps) % 32
 *   → identical for every lane in the warp → 32-way bank conflict.
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

/**
 * Run a kernel (function pointer), time it, and return average time in ms.
 *
 * @param kernel   Kernel function pointer
 * @param d_sink   Device output buffer (NUM_BLOCKS floats)
 * @param label    Human-readable label for printing
 * @return         Average kernel time in milliseconds
 */
typedef void (*KernelFn)(float *, int);

static double run_benchmark(KernelFn kernel, float *d_sink, const char *label)
{
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        hipLaunchKernelGGL(kernel, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE),
                           0, 0, d_sink, INNER_ITERS);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        hipLaunchKernelGGL(kernel, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE),
                           0, 0, d_sink, INNER_ITERS);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    double avg_ms = total_ms / TIMED_ITERS;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Bandwidth calculation
    // Per kernel launch: each thread does SMEM_FLOATS/BLOCK_SIZE iterations,
    // each reading + writing 4 bytes, across INNER_ITERS inner iterations,
    // across NUM_BLOCKS blocks.
    double bytes_per_launch = 2.0 * SMEM_FLOATS * sizeof(float)
                              * (double)INNER_ITERS * NUM_BLOCKS;
    double bw_GBs = bytes_per_launch / (avg_ms * 1e-3) / BYTES_PER_GB;

    printf("%-14s : %8.3f ms  %8.2f GB/s\n", label, avg_ms, bw_GBs);
    return avg_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    printf("Device        : %s (id=%d)\n", prop.name, device_id);
    printf("Shared mem/blk: %zu bytes\n",
           (size_t)prop.sharedMemPerBlock);
    printf("Block size    : %d threads\n", BLOCK_SIZE);
    printf("Grid size     : %d blocks\n", NUM_BLOCKS);
    printf("Inner iters   : %d\n", INNER_ITERS);
    printf("Warmup iters  : %d\n", WARMUP_ITERS);
    printf("Timed iters   : %d\n\n", TIMED_ITERS);

    // Allocate device sink buffer
    float *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, NUM_BLOCKS * sizeof(float)));

    // --- No-conflict ---
    double nc_ms = run_benchmark(smem_bw_no_conflict, d_sink, "no_conflict");

    // --- Conflict ---
    double c_ms  = run_benchmark(smem_bw_conflict, d_sink, "conflict");

    // Bandwidth numbers (recomputed for CSV)
    double bytes_per_launch = 2.0 * SMEM_FLOATS * sizeof(float)
                              * (double)INNER_ITERS * NUM_BLOCKS;
    double nc_GBs = bytes_per_launch / (nc_ms * 1e-3) / BYTES_PER_GB;
    double c_GBs  = bytes_per_launch / (c_ms  * 1e-3) / BYTES_PER_GB;

    // CSV output
    printf("shared_mem_bw,no_conflict,%.3f,%.2f\n", nc_ms, nc_GBs);
    printf("shared_mem_bw,conflict,%.3f,%.2f\n",    c_ms,  c_GBs);

    HIP_CHECK(hipFree(d_sink));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
