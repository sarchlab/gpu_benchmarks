// polybench_bicg.hip — PolyBench BiCG sub-kernel benchmark (HIP, self-contained)
//
// Computes:
//   s = A^T * r  (matrix-vector product with transpose)
//   q = A   * p  (matrix-vector product)
// where A is M×N, p is N-vector, r is M-vector,
//       s is N-vector (output), q is M-vector (output).
// Derived from the PolyBench/GPU benchmark suite.
//
// Two GPU kernels:
//   bicg_kernel1: s[j] = sum_i A[i,j] * r[i]   (one thread per column)
//   bicg_kernel2: q[i] = sum_j A[i,j] * p[j]   (one thread per row)
//
// Default: M = N = 4096
// Measures memory bandwidth in GB/s (A is read twice per iteration).
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./polybench_bicg [--m M] [--n N] [--iterations I]
//
//   --m M            Number of rows    (default: 4096)
//   --n N            Number of columns (default: 4096)
//   --iterations I   Timed iterations  (default: 5)
//
// Output (stdout): CSV — bicg,<N>,<time_ms>,<GBs>
// Output (stderr): device info, timing details

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Inlined utilities
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
// Kernel 1: s = A^T * r
// Each thread handles one column j: s[j] = sum_i A[i,j] * r[i]
// ---------------------------------------------------------------------------
__global__ void bicg_kernel1(const float* __restrict__ A,
                              const float* __restrict__ r,
                              float* __restrict__ s,
                              int m, int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < m; i++)
        sum += A[i * n + j] * r[i];
    s[j] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: q = A * p
// Each thread handles one row i: q[i] = sum_j A[i,j] * p[j]
// ---------------------------------------------------------------------------
__global__ void bicg_kernel2(const float* __restrict__ A,
                              const float* __restrict__ p,
                              float* __restrict__ q,
                              int m, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    float sum = 0.0f;
    for (int j = 0; j < n; j++)
        sum += A[i * n + j] * p[j];
    q[i] = sum;
}

// ---------------------------------------------------------------------------
// CPU reference for correctness check
// ---------------------------------------------------------------------------
static void cpu_bicg(const float* A, const float* p, const float* r,
                     float* s_ref, float* q_ref, int m, int n) {
    // s = A^T * r
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++)
            sum += A[i * n + j] * r[i];
        s_ref[j] = sum;
    }
    // q = A * p
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * p[j];
        q_ref[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int m     = parseIntParam(argc, argv, "--m", 4096);
    int n     = parseIntParam(argc, argv, "--n", 4096);
    int iters = parseIntParam(argc, argv, "--iterations", 5);

    size_t bytes_A = (size_t)m * n * sizeof(float);
    size_t bytes_p = (size_t)n * sizeof(float);
    size_t bytes_r = (size_t)m * sizeof(float);
    size_t bytes_s = (size_t)n * sizeof(float);
    size_t bytes_q = (size_t)m * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d  |  Iterations: %d\n\n", m, n, iters);

    // Host allocations
    float* h_A = (float*)malloc(bytes_A);
    float* h_p = (float*)malloc(bytes_p);
    float* h_r = (float*)malloc(bytes_r);
    float* h_s = (float*)malloc(bytes_s);
    float* h_q = (float*)malloc(bytes_q);

    // Initialize with random floats
    srand(42);
    for (int i = 0; i < m * n; ++i)
        h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int j = 0; j < n; ++j)
        h_p[j] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < m; ++i)
        h_r[i] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_p, *d_r, *d_s, *d_q;
    HIP_CHECK(hipMalloc(&d_A, bytes_A));
    HIP_CHECK(hipMalloc(&d_p, bytes_p));
    HIP_CHECK(hipMalloc(&d_r, bytes_r));
    HIP_CHECK(hipMalloc(&d_s, bytes_s));
    HIP_CHECK(hipMalloc(&d_q, bytes_q));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_p, h_p, bytes_p, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_r, h_r, bytes_r, hipMemcpyHostToDevice));

    int block_size = 256;
    int grid_k1 = (n + block_size - 1) / block_size;  // kernel1: one thread per column
    int grid_k2 = (m + block_size - 1) / block_size;  // kernel2: one thread per row

    BenchmarkTimer timer;

    // Warmup
    hipLaunchKernelGGL(bicg_kernel1, grid_k1, block_size, 0, 0, d_A, d_r, d_s, m, n);
    hipLaunchKernelGGL(bicg_kernel2, grid_k2, block_size, 0, 0, d_A, d_p, d_q, m, n);
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations
    std::vector<double> times(iters);
    for (int iter = 0; iter < iters; ++iter) {
        timer.record_start();
        hipLaunchKernelGGL(bicg_kernel1, grid_k1, block_size, 0, 0, d_A, d_r, d_s, m, n);
        hipLaunchKernelGGL(bicg_kernel2, grid_k2, block_size, 0, 0, d_A, d_p, d_q, m, n);
        timer.record_stop();
        times[iter] = static_cast<double>(timer.elapsed_ms());
    }

    // Copy results back
    HIP_CHECK(hipMemcpy(h_s, d_s, bytes_s, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_q, d_q, bytes_q, hipMemcpyDeviceToHost));

    // Correctness check against CPU reference (small subset)
    int check_n = (n < 8) ? n : 8;
    int check_m = (m < 8) ? m : 8;
    std::vector<float> s_ref(n), q_ref(m);
    cpu_bicg(h_A, h_p, h_r, s_ref.data(), q_ref.data(), m, n);

    bool pass = true;
    for (int j = 0; j < check_n; ++j) {
        float ref = s_ref[j];
        float rel = fabsf(h_s[j] - ref) / (fabsf(ref) + 1e-6f);
        if (rel > 1e-4f) {
            fprintf(stderr, "MISMATCH s[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    j, h_s[j], ref, rel);
            pass = false;
        }
    }
    for (int i = 0; i < check_m; ++i) {
        float ref = q_ref[i];
        float rel = fabsf(h_q[i] - ref) / (fabsf(ref) + 1e-6f);
        if (rel > 1e-4f) {
            fprintf(stderr, "MISMATCH q[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, h_q[i], ref, rel);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (first %d s, %d q elements): %s\n\n",
            check_n, check_m, pass ? "PASS" : "FAIL");

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

    // GB/s: A is read twice (kernel1 + kernel2), each is M*N*4 bytes
    double gb   = 2.0 * (double)m * (double)n * sizeof(float) / 1e9;
    double gbps = gb / (avg_ms * 1e-3);

    fprintf(stderr, "Bandwidth: %.2f GB/s  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gbps, avg_ms, mn, mx, stddev);

    // CSV output: bicg,<N>,<time_ms>,<GBs>
    printf("bicg,%d,%.4f,%.2f\n", n, avg_ms, gbps);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_p));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_s));
    HIP_CHECK(hipFree(d_q));
    free(h_A);
    free(h_p);
    free(h_r);
    free(h_s);
    free(h_q);

    return 0;
}
