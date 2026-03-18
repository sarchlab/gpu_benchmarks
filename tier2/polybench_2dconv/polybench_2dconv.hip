// polybench_2dconv.hip — PolyBench 2D Convolution benchmark (HIP, self-contained)
//
// Applies a fixed 3×3 kernel to an NI×NJ matrix A, producing output B.
// PolyBench standard kernel (c coefficients are fixed):
//
//   c = {{0.8, 0.2, 0.3},
//        {0.2, 0.7, 0.4},
//        {0.1, 0.2, 0.5}}
//
//   B[i][j] = sum_{k1=0}^{2} sum_{k2=0}^{2} c[k1][k2] * A[i+k1-1][j+k2-1]
//             for 1 <= i < NI-1, 1 <= j < NJ-1
//
// Each GPU thread computes one output element (one 3×3 dot product).
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./polybench_2dconv [--size N] [--iterations I]
//
//   --size N        Matrix dimension N×N   (default: 2048)
//   --iterations I  Timed benchmark iterations (default: 5)
//
// Output (stdout): CSV — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
// Output (stderr): GFLOPS

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
// PolyBench 2D Convolution kernel
// ---------------------------------------------------------------------------
// Fixed 3×3 kernel coefficients (PolyBench standard):
//   c[0][0]=0.8  c[0][1]=0.2  c[0][2]=0.3
//   c[1][0]=0.2  c[1][1]=0.7  c[1][2]=0.4
//   c[2][0]=0.1  c[2][1]=0.2  c[2][2]=0.5
//
// Border elements (row 0, row N-1, col 0, col M-1) are unchanged (left as-is
// in B — threads simply skip them, matching PolyBench reference).

__global__ void convolution2D_kernel(const float* __restrict__ A,
                                     float*       __restrict__ B,
                                     int NI, int NJ) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (i < 1 || i >= NI - 1 || j < 1 || j >= NJ - 1)
        return;

    // PolyBench fixed coefficients
    const float c00 = 0.8f, c01 = 0.2f, c02 = 0.3f;
    const float c10 = 0.2f, c11 = 0.7f, c12 = 0.4f;
    const float c20 = 0.1f, c21 = 0.2f, c22 = 0.5f;

    B[i * NJ + j] =
        c00 * A[(i - 1) * NJ + (j - 1)] +
        c01 * A[(i - 1) * NJ +  j     ] +
        c02 * A[(i - 1) * NJ + (j + 1)] +
        c10 * A[ i      * NJ + (j - 1)] +
        c11 * A[ i      * NJ +  j     ] +
        c12 * A[ i      * NJ + (j + 1)] +
        c20 * A[(i + 1) * NJ + (j - 1)] +
        c21 * A[(i + 1) * NJ +  j     ] +
        c22 * A[(i + 1) * NJ + (j + 1)];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int N    = parseIntParam(argc, argv, "--size", 2048);
    int iters = parseIterations(argc, argv);

    int NI = N, NJ = N;
    size_t matSize = (size_t)NI * NJ * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix: %d×%d  |  Iterations: %d\n\n", NI, NJ, iters);

    // Host allocations
    float* h_A = (float*)malloc(matSize);
    if (!h_A) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(42);
    for (int i = 0; i < NI * NJ; i++)
        h_A[i] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, matSize));
    HIP_CHECK(hipMalloc(&d_B, matSize));

    HIP_CHECK(hipMemcpy(d_A, h_A, matSize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B, 0, matSize));

    dim3 block(16, 16);
    dim3 grid((NJ + block.x - 1) / block.x, (NI + block.y - 1) / block.y);

    char problemSize[64];
    snprintf(problemSize, sizeof(problemSize), "%d", N);

    printCSVHeader();

    BenchResult r = runBenchmark("polybench_2dconv", problemSize, iters, [&]() {
        hipLaunchKernelGGL(convolution2D_kernel, grid, block, 0, 0,
                           d_A, d_B, NI, NJ);
        HIP_CHECK(hipDeviceSynchronize());
    });

    printCSVRow(r);

    // GFLOPS: each interior output element = 9 MACs = 18 FLOPs
    long long interior = (long long)(NI - 2) * (NJ - 2);
    double gflops = 2.0 * 9.0 * (double)interior / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "GFLOPS: %.2f  (avg %.4f ms)\n", gflops, r.avg_ms);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_A);

    return 0;
}
