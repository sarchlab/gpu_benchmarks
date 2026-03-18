// shoc_sort.hip — SHOC Sort benchmark (HIP, self-contained)
//
// Bitonic sort on a large array of random unsigned integers.
// N = 1<<22 (4M elements) by default.
//
// Algorithm: bitonic sort — O(n log^2 n) compare-and-swap network.
// Each GPU kernel pass performs one compare-and-swap step.
//
// Compiles with hipcc (AMD ROCm) or nvcc (CUDA via HIP compat layer).
//
// Usage:
//   ./shoc_sort [--size N] [--iterations I]
//
//   --size N         Number of elements (must be power of 2, default: 4194304)
//   --iterations I   Timed iterations (default: 5)
//
// Output (stdout): CSV — shoc_sort,<N>,<Melements/s>
// Output (stderr): device info, timing details

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Inlined HIP utilities (no shared headers)
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

static int parseIterations(int argc, char** argv) {
    int iters = 5;
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

// ---------------------------------------------------------------------------
// Bitonic sort kernel
// Each thread performs one compare-and-swap in the bitonic network.
// Parameters:
//   data  — array to sort (in-place)
//   j     — current sub-step distance
//   k     — current stage size (power of 2)
// ---------------------------------------------------------------------------
__global__ void bitonic_step(unsigned int* data, int j, int k) {
    int i = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    int ixj = i ^ j;

    if (ixj > i) {
        // Determine sort direction: ascending when (i & k) == 0
        bool ascending = ((i & k) == 0);
        unsigned int a = data[i];
        unsigned int b = data[ixj];
        if (ascending ? (a > b) : (a < b)) {
            data[i]   = b;
            data[ixj] = a;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU bitonic sort driver
// Launches O(log^2 N) kernel passes.
// N must be a power of 2.
// ---------------------------------------------------------------------------
static void bitonic_sort_gpu(unsigned int* d_data, int N, int block_size) {
    int grid = (N + block_size - 1) / block_size;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            hipLaunchKernelGGL(bitonic_step, dim3(grid), dim3(block_size),
                               0, 0, d_data, j, k);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Verify: check array is sorted
// ---------------------------------------------------------------------------
static bool verify_sorted(const unsigned int* arr, int N) {
    for (int i = 1; i < N; ++i) {
        if (arr[i] < arr[i - 1]) {
            fprintf(stderr, "Mismatch at %d: arr[%d]=%u > arr[%d]=%u\n",
                    i, i - 1, arr[i - 1], i, arr[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int N     = parseIntParam(argc, argv, "--size", 1 << 22); // 4M default
    int iters = parseIterations(argc, argv);

    // Round N up to next power of 2
    {
        int p = 1;
        while (p < N) p <<= 1;
        if (p != N) {
            fprintf(stderr, "Warning: N=%d is not a power of 2, rounding up to %d\n", N, p);
            N = p;
        }
    }

    const int BLOCK_SIZE = 256;
    size_t bytes = (size_t)N * sizeof(unsigned int);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d elements (%.1f MB) | Iterations: %d\n\n",
            N, (double)bytes / (1024.0 * 1024.0), iters);

    // Host allocation and initialization
    unsigned int* h_data = (unsigned int*)malloc(bytes);
    unsigned int* h_orig = (unsigned int*)malloc(bytes);
    if (!h_data || !h_orig) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_orig[i] = (unsigned int)rand();
    }

    // Device allocation
    unsigned int* d_data;
    HIP_CHECK(hipMalloc(&d_data, bytes));

    BenchmarkTimer timer;

    // Warmup (not measured)
    HIP_CHECK(hipMemcpy(d_data, h_orig, bytes, hipMemcpyHostToDevice));
    bitonic_sort_gpu(d_data, N, BLOCK_SIZE);

    // Timed iterations
    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        // Fresh input each iteration
        HIP_CHECK(hipMemcpy(d_data, h_orig, bytes, hipMemcpyHostToDevice));
        timer.record_start();
        bitonic_sort_gpu(d_data, N, BLOCK_SIZE);
        timer.record_stop();
        times[i] = static_cast<double>(timer.elapsed_ms());
        fprintf(stderr, "  Iter %d: %.4f ms\n", i + 1, times[i]);
    }

    // Statistics
    double sum_ms = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iters; ++i) {
        sum_ms += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg_ms = sum_ms / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; ++i) {
        double d = times[i] - avg_ms;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;

    double melems_per_sec = (double)N / (avg_ms * 1e-3) / 1e6;

    fprintf(stderr, "Throughput: %.2f Melements/s  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            melems_per_sec, avg_ms, mn, mx, stddev);

    // Correctness check (last iteration result still in d_data)
    HIP_CHECK(hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost));
    if (verify_sorted(h_data, N)) {
        fprintf(stderr, "Correctness: PASS\n");
    } else {
        fprintf(stderr, "Correctness: FAIL\n");
    }

    // CSV output: shoc_sort,<N>,<Melements/s>
    printf("shoc_sort,%d,%.2f\n", N, melems_per_sec);

    // Cleanup
    HIP_CHECK(hipFree(d_data));
    free(h_data);
    free(h_orig);

    return 0;
}
