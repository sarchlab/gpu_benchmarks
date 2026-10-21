// polybench_bicg.cu — PolyBench BiCG sub-kernel benchmark (CUDA, self-contained)
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
// Native CUDA port of ../hip/polybench_bicg.hip; compiles with nvcc.
//
// Usage:
//   ./polybench_bicg [--m M] [--n N]
//
//   --m M            Number of rows    (default: 4096)
//   --n N            Number of columns (default: 4096)
//
// Environment variables (override argv):
//   BENCH_PARAM_m, BENCH_PARAM_n, BENCH_PARAM_block_size
//
// Output (stdout): the BiCG pass runs exactly once; one {"type":"kernel",...}
// JSON line per kernel (bicg_kernel1, bicg_kernel2) carrying that kernel's
// GPU event time (time_ms) and host time (host_time_ms), then a summary whose
// total_time_ms is the SUM of the two GPU event times. Bandwidth (GB/s) is
// derived downstream from time_ms and params.
// Output (stderr): device info, timing details

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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
#include "cpu_reference.h"

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

    size_t bytes_A = (size_t)m * n * sizeof(float);
    size_t bytes_p = (size_t)n * sizeof(float);
    size_t bytes_r = (size_t)m * sizeof(float);
    size_t bytes_s = (size_t)n * sizeof(float);
    size_t bytes_q = (size_t)m * sizeof(float);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
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
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_p, bytes_p));
    CUDA_CHECK(cudaMalloc(&d_r, bytes_r));
    CUDA_CHECK(cudaMalloc(&d_s, bytes_s));
    CUDA_CHECK(cudaMalloc(&d_q, bytes_q));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, h_p, bytes_p, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r, bytes_r, cudaMemcpyHostToDevice));

    int grid_k1 = (n + block_size - 1) / block_size;  // kernel1: one thread per column
    int grid_k2 = (m + block_size - 1) / block_size;  // kernel2: one thread per row

    char params[192];
    snprintf(params, sizeof(params),
             "\"m\":%d,\"n\":%d,\"block_size\":%d,\"verification\":\"full_cpu_v1\"", m, n, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2;

    // One full BiCG pass (run exactly once): both matrix-vector products,
    // each timed separately. No reset is needed: inputs A/r/p are read-only
    // and the outputs s/q are fully overwritten, so every execution does
    // identical work.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_k1);
        bench_lap_reset(&acc_k2);

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
            bicg_kernel1<<<grid_k1, block_size>>>(d_A, d_r, d_s, m, n);
        }));
        bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
            bicg_kernel2<<<grid_k2, block_size>>>(d_A, d_p, d_q, m, n);
        }));
    };

    // Single timed pass — one record per kernel
    run_pass();
    double total_gpu_ms = 0.0;
    bench_emit_iter("bicg_kernel1", 0, &acc_k1, params);
    bench_emit_iter("bicg_kernel2", 0, &acc_k2, params);
    total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms;
    fprintf(stderr, "bicg_kernel1 gpu = %.3f ms, bicg_kernel2 gpu = %.3f ms\n",
            acc_k1.gpu_ms, acc_k2.gpu_ms);

    bench_timer_destroy(&timer);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_s, d_s, bytes_s, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_q, d_q, bytes_q, cudaMemcpyDeviceToHost));

    // Correctness check against full CPU reference
    std::vector<float> s_ref(n), q_ref(m);
    cpu_bicg(h_A, h_p, h_r, s_ref.data(), q_ref.data(), m, n);

    bool pass = true;
    for (int j = 0; j < n; ++j) {
        float ref = s_ref[j];
        float rel = fabsf(h_s[j] - ref) / (fabsf(ref) + 1e-6f);
        if (!std::isfinite(h_s[j]) || !std::isfinite(ref) || !std::isfinite(rel) ||
            rel > 1e-4f) {
            fprintf(stderr, "MISMATCH s[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    j, h_s[j], ref, rel);
            pass = false;
        }
    }
    for (int i = 0; i < m; ++i) {
        float ref = q_ref[i];
        float rel = fabsf(h_q[i] - ref) / (fabsf(ref) + 1e-6f);
        if (!std::isfinite(h_q[i]) || !std::isfinite(ref) || !std::isfinite(rel) ||
            rel > 1e-4f) {
            fprintf(stderr, "MISMATCH q[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, h_q[i], ref, rel);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (all %d s, %d q elements): %s\n\n",
            n, m, pass ? "PASS" : "FAIL");

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_q));
    free(h_A);
    free(h_p);
    free(h_r);
    free(h_s);
    free(h_q);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
