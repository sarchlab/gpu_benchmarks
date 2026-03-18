// shoc_reduction.hip — SHOC Reduction benchmark (HIP, self-contained)
//
// Parallel sum reduction: reduces an array of floats to a single sum using
// tree-based shared-memory reduction. Derived from the SHOC benchmark suite.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./shoc_reduction [--size N] [--block_size B] [--iterations I]
//
//   --size N         Number of float elements (default: 4194304 = 4 M)
//   --block_size B   GPU thread-block size    (default: 256)
//   --iterations I   Timed iterations         (default: 20)
//
// Output (stdout): CSV row — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
// Output (stderr): Effective bandwidth in GB/s (2 * N * sizeof(float) / time_sec), PASS/FAIL

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Inlined bench_common_hip.h utilities
// ---------------------------------------------------------------------------

// Error-checking macro
#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Result record
struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

// Parse --iterations N from argv (default: 20)
static int parseIterations(int argc, char** argv) {
    int iters = 20;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--iterations") == 0) {
            int v = atoi(argv[i + 1]);
            if (v >= 1) iters = v;
            break;
        }
    }
    return iters;
}

// Parse --<name> N from argv (returns defaultVal if not found)
static int parseIntParam(int argc, char** argv, const char* name,
                         int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

// CSV output helpers
static void printCSVHeader() {
    printf("kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms\n");
}

static void printCSVRow(const BenchResult& r) {
    printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
           r.kernel_name, r.problem_size, r.iterations,
           r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

// HIP-event-based timer
struct BenchmarkTimer {
    hipEvent_t start, stop;
    BenchmarkTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~BenchmarkTimer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    void record_start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start, stream));
    }
    void record_stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// runBenchmark: one warmup + N timed iterations, returns stats
template <typename Func>
static BenchResult runBenchmark(const char* kernel_name,
                                const char* problem_size,
                                int         iterations,
                                Func        func) {
    BenchmarkTimer timer;

    // Warmup (not measured)
    func();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        timer.record_start();
        func();
        timer.record_stop();
        times[i] = static_cast<double>(timer.elapsed_ms());
    }

    double sum = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum / iterations;

    double variance = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iterations > 1) ? sqrt(variance / (iterations - 1)) : 0.0;

    BenchResult r;
    r.kernel_name  = kernel_name;
    r.problem_size = problem_size;
    r.iterations   = iterations;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;
    return r;
}

// ---------------------------------------------------------------------------
// Reduction kernel — shared-memory tree reduction
//
// Each block reduces 2*blockDim.x elements to one partial sum.
// The partial sums are written to output[blockIdx.x].
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads and sums two elements (with bounds check)
    float val = 0.0f;
    if (i < (unsigned int)N)
        val = input[i];
    if (i + blockDim.x < (unsigned int)N)
        val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the block's partial sum
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Two-pass (multi-level if needed) GPU reduction
//
// Reduces d_input[0..N-1] to a single float at d_result[0].
// d_partial must be at least num_blocks floats in size.
// ---------------------------------------------------------------------------
static void gpu_reduce(const float* d_input, float* d_partial,
                       float* d_result, int N, int block_size)
{
    int elems_per_block = block_size * 2;
    int num_blocks      = (N + elems_per_block - 1) / elems_per_block;
    int shared_bytes    = block_size * sizeof(float);

    // Pass 1: reduce input → partial sums (one per block)
    hipLaunchKernelGGL(reduce_kernel, dim3(num_blocks), dim3(block_size),
                       shared_bytes, 0,
                       d_input, d_partial, N);

    if (num_blocks == 1) {
        // Already finished — copy d_partial[0] → d_result[0]
        HIP_CHECK(hipMemcpy(d_result, d_partial, sizeof(float),
                             hipMemcpyDeviceToDevice));
        return;
    }

    // Pass 2 (and beyond if needed): keep reducing until one value remains
    // For typical problem sizes num_blocks is a few thousand at most,
    // so two passes are sufficient. Use recursive approach for generality.
    int remaining = num_blocks;
    float* d_src  = d_partial;   // source for next pass
    float* d_dst  = nullptr;

    // Allocate a temporary buffer for intermediate passes
    float* d_temp = nullptr;
    bool   allocated_temp = false;

    while (remaining > 1) {
        int next_blocks  = (remaining + elems_per_block - 1) / elems_per_block;

        if (next_blocks == 1) {
            // Final pass writes to d_result
            hipLaunchKernelGGL(reduce_kernel, dim3(1), dim3(block_size),
                               shared_bytes, 0,
                               d_src, d_result, remaining);
            remaining = 1;
        } else {
            // Intermediate pass — need a temp buffer
            if (!allocated_temp) {
                HIP_CHECK(hipMalloc(&d_temp, next_blocks * sizeof(float)));
                allocated_temp = true;
            }
            hipLaunchKernelGGL(reduce_kernel, dim3(next_blocks), dim3(block_size),
                               shared_bytes, 0,
                               d_src, d_temp, remaining);
            d_src     = d_temp;
            remaining = next_blocks;
        }
    }

    if (allocated_temp) {
        HIP_CHECK(hipFree(d_temp));
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
static double reduction_cpu(const float* data, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int N          = parseIntParam(argc, argv, "--size", 4194304);
    int block_size = parseIntParam(argc, argv, "--block_size", 256);
    int iters      = parseIterations(argc, argv);

    size_t bytesIn = (size_t)N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d floats (%.1f MiB)\n",
            N, (double)bytesIn / (1024.0 * 1024.0));
    fprintf(stderr, "Block size: %d  |  Iterations: %d\n\n", block_size, iters);

    // Host allocation
    float* h_input = (float*)malloc(bytesIn);

    // Initialize with a known pattern
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i % 1000) * 0.001f;
    }

    // Device allocations
    int elems_per_block = block_size * 2;
    int num_blocks      = (N + elems_per_block - 1) / elems_per_block;

    float *d_input, *d_partial, *d_result;
    HIP_CHECK(hipMalloc(&d_input,   bytesIn));
    HIP_CHECK(hipMalloc(&d_partial, (size_t)num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_result,  sizeof(float)));

    HIP_CHECK(hipMemcpy(d_input, h_input, bytesIn, hipMemcpyHostToDevice));

    // Benchmark
    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%d", N);

    printCSVHeader();

    BenchResult r = runBenchmark("reduction", size_str, iters, [&]() {
        gpu_reduce(d_input, d_partial, d_result, N, block_size);
    });

    printCSVRow(r);

    // Effective bandwidth: 2 * N * sizeof(float) (read + result write)
    double bandwidth_gb = 2.0 * (double)N * sizeof(float)
                          / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Effective bandwidth: %.2f GB/s  (avg %.4f ms)\n",
            bandwidth_gb, r.avg_ms);

    // Verify
    float gpu_result = 0.0f;
    HIP_CHECK(hipMemcpy(&gpu_result, d_result, sizeof(float),
                         hipMemcpyDeviceToHost));
    double cpu_result = reduction_cpu(h_input, N);

    double rel_err = fabs((double)gpu_result - cpu_result)
                     / (fabs(cpu_result) + 1e-10);
    fprintf(stderr, "GPU sum: %f, CPU sum: %f, relative error: %e\n",
            gpu_result, cpu_result, rel_err);

    if (rel_err < 1e-3) {
        fprintf(stderr, "PASS\n");
    } else {
        fprintf(stderr, "FAIL: relative error too large\n");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_partial));
    HIP_CHECK(hipFree(d_result));
    free(h_input);

    return (rel_err >= 1e-3) ? 1 : 0;
}
