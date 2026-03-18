// polybench_3mm.hip — PolyBench 3mm benchmark (HIP, self-contained)
//
// Three chained matrix multiplications:
//   E = A * B  (NI×NK · NK×NJ → NI×NJ)
//   F = C * D  (NJ×NM · NM×NL → NJ×NL)
//   G = E * F  (NI×NJ · NJ×NL → NI×NL)
//
// Default matrix sizes: NI=NJ=NK=NL=NM=256 (all square 256×256)
// Reports GFLOPS = total_flops / time_s / 1e9
//   total_flops = 2*NI*NK*NJ + 2*NJ*NM*NL + 2*NI*NJ*NL
//
// Usage:
//   ./polybench_3mm [--size N] [--iterations I]
//
//   --size N         Matrix dimension N (NxN for all matrices) (default: 256)
//   --iterations I   Timed iterations                          (default: 5)
//
// Output (stdout): CSV — polybench_3mm,<N>,<GFLOPS>
// Output (stderr): device info, timing details

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
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

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

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
// GPU Kernels
// Each thread computes one output element (simple dot-product loop).
// Block: 16×16, Grid: ceil(cols/16) × ceil(rows/16)
// ---------------------------------------------------------------------------

// Kernel 1: E = A * B   (A: NI×NK, B: NK×NJ → E: NI×NJ)
__global__ void mm3_kernel1(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__       E,
                             int NI, int NK, int NJ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && j < NJ) {
        float sum = 0.0f;
        for (int k = 0; k < NK; k++)
            sum += A[i * NK + k] * B[k * NJ + j];
        E[i * NJ + j] = sum;
    }
}

// Kernel 2: F = C * D   (C: NJ×NM, D: NM×NL → F: NJ×NL)
__global__ void mm3_kernel2(const float* __restrict__ C,
                             const float* __restrict__ D,
                             float* __restrict__       F,
                             int NJ, int NM, int NL) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < NJ && l < NL) {
        float sum = 0.0f;
        for (int m = 0; m < NM; m++)
            sum += C[j * NM + m] * D[m * NL + l];
        F[j * NL + l] = sum;
    }
}

// Kernel 3: G = E * F   (E: NI×NJ, F: NJ×NL → G: NI×NL)
__global__ void mm3_kernel3(const float* __restrict__ E,
                             const float* __restrict__ F,
                             float* __restrict__       G,
                             int NI, int NJ, int NL) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && l < NL) {
        float sum = 0.0f;
        for (int j = 0; j < NJ; j++)
            sum += E[i * NJ + j] * F[j * NL + l];
        G[i * NL + l] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int N     = parseIntParam(argc, argv, "--size", 256);
    int iters = parseIterations(argc, argv);

    // All matrices are N×N
    const int NI = N, NJ = N, NK = N, NL = N, NM = N;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d (all)  |  Iterations: %d\n\n",
            N, N, iters);

    // Buffer sizes
    size_t sA = (size_t)NI * NK * sizeof(float); // A: NI×NK
    size_t sB = (size_t)NK * NJ * sizeof(float); // B: NK×NJ
    size_t sC = (size_t)NJ * NM * sizeof(float); // C: NJ×NM
    size_t sD = (size_t)NM * NL * sizeof(float); // D: NM×NL
    size_t sE = (size_t)NI * NJ * sizeof(float); // E: NI×NJ
    size_t sF = (size_t)NJ * NL * sizeof(float); // F: NJ×NL
    size_t sG = (size_t)NI * NL * sizeof(float); // G: NI×NL

    // Host allocations — inputs only (A, B, C, D)
    float* h_A = (float*)malloc(sA);
    float* h_B = (float*)malloc(sB);
    float* h_C = (float*)malloc(sC);
    float* h_D = (float*)malloc(sD);

    // Initialize with random floats
    srand(42);
    for (int i = 0; i < NI * NK; ++i) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NK * NJ; ++i) h_B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NJ * NM; ++i) h_C[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NM * NL; ++i) h_D[i] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_G;
    HIP_CHECK(hipMalloc(&d_A, sA));
    HIP_CHECK(hipMalloc(&d_B, sB));
    HIP_CHECK(hipMalloc(&d_C, sC));
    HIP_CHECK(hipMalloc(&d_D, sD));
    HIP_CHECK(hipMalloc(&d_E, sE));
    HIP_CHECK(hipMalloc(&d_F, sF));
    HIP_CHECK(hipMalloc(&d_G, sG));

    HIP_CHECK(hipMemcpy(d_A, h_A, sA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, sB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, sC, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_D, h_D, sD, hipMemcpyHostToDevice));

    // Kernel launch configuration (16×16 blocks)
    dim3 block(16, 16);
    // Kernel 1: E = A*B  grid over (NJ, NI)
    dim3 grid1((NJ + 15) / 16, (NI + 15) / 16);
    // Kernel 2: F = C*D  grid over (NL, NJ)
    dim3 grid2((NL + 15) / 16, (NJ + 15) / 16);
    // Kernel 3: G = E*F  grid over (NL, NI)
    dim3 grid3((NL + 15) / 16, (NI + 15) / 16);

    BenchmarkTimer timer;

    // Warmup (not measured)
    hipLaunchKernelGGL(mm3_kernel1, grid1, block, 0, 0, d_A, d_B, d_E, NI, NK, NJ);
    hipLaunchKernelGGL(mm3_kernel2, grid2, block, 0, 0, d_C, d_D, d_F, NJ, NM, NL);
    hipLaunchKernelGGL(mm3_kernel3, grid3, block, 0, 0, d_E, d_F, d_G, NI, NJ, NL);
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations — all 3 kernels per iteration
    std::vector<double> times(iters);
    for (int iter = 0; iter < iters; ++iter) {
        timer.record_start();
        hipLaunchKernelGGL(mm3_kernel1, grid1, block, 0, 0, d_A, d_B, d_E, NI, NK, NJ);
        hipLaunchKernelGGL(mm3_kernel2, grid2, block, 0, 0, d_C, d_D, d_F, NJ, NM, NL);
        hipLaunchKernelGGL(mm3_kernel3, grid3, block, 0, 0, d_E, d_F, d_G, NI, NJ, NL);
        timer.record_stop();
        times[iter] = static_cast<double>(timer.elapsed_ms());
    }

    // Compute statistics
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

    // total_flops = 2*NI*NK*NJ + 2*NJ*NM*NL + 2*NI*NJ*NL
    double total_flops = 2.0 * ((double)NI * NK * NJ +
                                (double)NJ * NM * NL +
                                (double)NI * NJ * NL);
    double gflops = total_flops / (avg_ms * 1e-3) / 1e9;

    fprintf(stderr, "Performance: %.2f GFLOPS  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gflops, avg_ms, mn, mx, stddev);

    // CSV output: polybench_3mm,<N>,<GFLOPS>
    printf("polybench_3mm,%d,%.2f\n", N, gflops);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_E));
    HIP_CHECK(hipFree(d_F));
    HIP_CHECK(hipFree(d_G));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}
