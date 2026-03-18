// shoc_devicememory.hip — SHOC DeviceMemory benchmark (HIP, self-contained)
//
// Device global memory bandwidth benchmark: read-only, write-only, and
// read-write patterns.  Measures GB/s for each access pattern.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./shoc_devicememory [--array_size_mb N] [--block_size B] [--iterations I]
//
//   --array_size_mb N  Array size in megabytes  (default: 64)
//   --block_size B     Thread-block size        (default: 256)
//   --iterations I     Timed iterations         (default: 20)
//
// Output (stdout): 3 CSV rows (readOnly, writeOnly, readWrite)
// Output (stderr): bandwidth in GB/s for each pattern

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

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

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

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

static void printCSVHeader() {
    printf("kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms\n");
}

static void printCSVRow(const BenchResult& r) {
    printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
           r.kernel_name, r.problem_size, r.iterations,
           r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

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

template <typename Func>
static BenchResult runBenchmark(const char* kernel_name,
                                const char* problem_size,
                                int         iterations,
                                Func        func) {
    BenchmarkTimer timer;

    // Warmup
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
// Kernels
// ---------------------------------------------------------------------------

// Read-only: read elements from input, accumulate, write single result per thread
__global__ void readOnly_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = gid; i < N; i += totalThreads) {
        sum += input[i];
    }
    // Prevent dead-code elimination
    if (gid < N) {
        output[gid] = sum;
    }
}

// Write-only: write a value to each element (strided access pattern)
__global__ void writeOnly_kernel(float* __restrict__ output,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        output[i] = (float)i * 0.001f;
    }
}

// Read-write: read + scale + write (triad-like, but with same array)
__global__ void readWrite_kernel(float* __restrict__ data,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        data[i] = data[i] * 1.0001f;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int array_size_mb = parseIntParam(argc, argv, "--array_size_mb", 64);
    int block_size    = parseIntParam(argc, argv, "--block_size", 256);
    int iters         = parseIterations(argc, argv);

    size_t N     = (size_t)array_size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d MB (%zu floats)  |  Block size: %d  |  Iterations: %d\n\n",
            array_size_mb, N, block_size, iters);

    // Device allocations
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input,  bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));

    // Initialize device input
    float* h_init = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i)
        h_init[i] = (float)(i % 1000) * 0.001f;
    HIP_CHECK(hipMemcpy(d_input, h_init, bytes, hipMemcpyHostToDevice));
    free(h_init);

    // Grid size: cap to avoid excessive launch overhead
    int num_blocks = (int)((N + block_size - 1) / block_size);
    if (num_blocks > 65535) num_blocks = 65535;

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dMB", array_size_mb);

    printCSVHeader();

    // --- Read-only bandwidth ---
    BenchResult r_read = runBenchmark("DeviceMemory_Read", size_str, iters, [&]() {
        hipLaunchKernelGGL(readOnly_kernel, dim3(num_blocks), dim3(block_size),
                           0, 0, d_input, d_output, (int)N);
    });
    printCSVRow(r_read);
    double read_bw = (double)bytes / (r_read.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Read-only bandwidth:  %.2f GB/s  (%d MB)\n", read_bw, array_size_mb);

    // --- Write-only bandwidth ---
    BenchResult r_write = runBenchmark("DeviceMemory_Write", size_str, iters, [&]() {
        hipLaunchKernelGGL(writeOnly_kernel, dim3(num_blocks), dim3(block_size),
                           0, 0, d_output, (int)N);
    });
    printCSVRow(r_write);
    double write_bw = (double)bytes / (r_write.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Write-only bandwidth: %.2f GB/s  (%d MB)\n", write_bw, array_size_mb);

    // --- Read-write bandwidth ---
    // Re-initialize d_input before read-write test
    {
        float* h_rw = (float*)malloc(bytes);
        for (size_t i = 0; i < N; ++i) h_rw[i] = (float)(i % 1000) * 0.001f;
        HIP_CHECK(hipMemcpy(d_input, h_rw, bytes, hipMemcpyHostToDevice));
        free(h_rw);
    }
    BenchResult r_rw = runBenchmark("DeviceMemory_ReadWrite", size_str, iters, [&]() {
        hipLaunchKernelGGL(readWrite_kernel, dim3(num_blocks), dim3(block_size),
                           0, 0, d_input, (int)N);
    });
    printCSVRow(r_rw);
    // Read-write moves 2× the array (1 read + 1 write)
    double rw_bw = 2.0 * (double)bytes / (r_rw.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Read-write bandwidth: %.2f GB/s  (%d MB)\n", rw_bw, array_size_mb);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    return 0;
}
