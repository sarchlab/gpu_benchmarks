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
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./polybench_bicg [--m M] [--n N]
//
//   --m M            Number of rows    (default: 4096)
//   --n N            Number of columns (default: 4096)
//
// Environment variables (override argv):
//   BENCH_PARAM_m, BENCH_PARAM_n, BENCH_PARAM_block_size
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (bicg_kernel1, bicg_kernel2) carrying that kernel's GPU event time
// (time_ms) and host time (host_time_ms), then a summary whose total_time_ms
// is the SUM of all recorded GPU event times. Bandwidth (GB/s) is derived
// downstream from time_ms and params.
// Output (stderr): device info, timing details

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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
    setvbuf(stdout, NULL, _IONBF, 0);

    int m     = parseIntParam(argc, argv, "--m", 4096);
    int n     = parseIntParam(argc, argv, "--n", 4096);
    int block_size = 256;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_m");
    if (env_val) m = atoi(env_val);
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) n = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

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
    fprintf(stderr, "Matrix size: %d×%d\n\n", m, n);

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

    int grid_k1 = (n + block_size - 1) / block_size;  // kernel1: one thread per column
    int grid_k2 = (m + block_size - 1) / block_size;  // kernel2: one thread per row

    char params[128];
    snprintf(params, sizeof(params),
             "\"m\":%d,\"n\":%d,\"block_size\":%d", m, n, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2;

    // One full BiCG pass: both matrix-vector products, each timed separately.
    // No per-pass reset is needed: inputs A/r/p are read-only and the outputs
    // s/q are fully overwritten every pass, so repeated passes do identical
    // work.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_k1);
        bench_lap_reset(&acc_k2);

        bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
            bicg_kernel1<<<grid_k1, block_size>>>(d_A, d_r, d_s, m, n);
        }));
        bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
            bicg_kernel2<<<grid_k2, block_size>>>(d_A, d_p, d_q, m, n);
        }));
    };

    // Warmup passes (records not emitted)
    for (int w = 0; w < num_warmup; ++w) {
        run_pass();
    }

    // Timed passes — one record per kernel per pass
    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_pass();
        bench_emit_iter("bicg_kernel1", i, &acc_k1, params);
        bench_emit_iter("bicg_kernel2", i, &acc_k2, params);
        total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms;
        fprintf(stderr, "pass %d: bicg_kernel1 gpu = %.3f ms, bicg_kernel2 gpu = %.3f ms\n",
                i, acc_k1.gpu_ms, acc_k2.gpu_ms);
    }

    bench_timer_destroy(&timer);

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

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

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
