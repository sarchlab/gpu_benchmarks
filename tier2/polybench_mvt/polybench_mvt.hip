// polybench_mvt.hip — PolyBench MVT benchmark (HIP, self-contained)
//
// Computes:
//   x1 = A * y1   (matrix-vector product)
//   x2 = A^T * y2 (transpose matrix-vector product)
// where A is N×N, x1, x2, y1, y2 are N-vectors.
//
// Derived from the PolyBench/GPU benchmark suite.
//
// Two kernels:
//   mvt_kernel1: x1[i] = sum_j A[i,j] * y1[j]   (one row per thread)
//   mvt_kernel2: x2[j] = sum_i A[i,j] * y2[i]   (one column per thread)
//
// Default: N=4096
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./polybench_mvt [--n N]
//
//   --n N            Matrix dimension (default: 4096)
//
// Environment variables (override argv):
//   BENCH_PARAM_n
//
// Output (stdout): the MVT pass runs exactly once; one {"type":"kernel",...}
// JSON line per kernel (mvt_kernel1, mvt_kernel2) carrying that kernel's GPU
// event time and host wall-clock time, then a summary whose total_time_ms is
// the SUM of the two GPU event times. Memory bandwidth (GB/s) is derived
// downstream from time_ms and params.
// Output (stderr): device info, timing, correctness check

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
// Kernel 1: x1 = A * y1
// Each thread computes one row: x1[i] = sum_j A[i,j] * y1[j]
// ---------------------------------------------------------------------------
__global__ void mvt_kernel1(const float* __restrict__ A,
                              const float* __restrict__ y1,
                              float* __restrict__ x1,
                              int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    for (int j = 0; j < n; j++)
        sum += A[i * n + j] * y1[j];
    x1[i] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: x2 = A^T * y2
// Each thread computes one column: x2[j] = sum_i A[i,j] * y2[i]
// ---------------------------------------------------------------------------
__global__ void mvt_kernel2(const float* __restrict__ A,
                              const float* __restrict__ y2,
                              float* __restrict__ x2,
                              int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += A[i * n + j] * y2[i];
    x2[j] = sum;
}

// ---------------------------------------------------------------------------
// CPU reference for correctness check
// ---------------------------------------------------------------------------
static void cpu_mvt(const float* A, const float* y1, const float* y2,
                    float* x1_ref, float* x2_ref, int n) {
    // x1 = A * y1
    for (int i = 0; i < n; i++) {
        x1_ref[i] = 0.0f;
        for (int j = 0; j < n; j++)
            x1_ref[i] += A[i * n + j] * y1[j];
    }
    // x2 = A^T * y2
    for (int j = 0; j < n; j++) {
        x2_ref[j] = 0.0f;
        for (int i = 0; i < n; i++)
            x2_ref[j] += A[i * n + j] * y2[i];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int n     = parseIntParam(argc, argv, "--n", 4096);
    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) n = atoi(env_val);

    size_t bytes_A  = (size_t)n * n * sizeof(float);
    size_t bytes_v  = (size_t)n * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d\n\n", n, n);

    // Host allocations
    float* h_A  = (float*)malloc(bytes_A);
    float* h_y1 = (float*)malloc(bytes_v);
    float* h_y2 = (float*)malloc(bytes_v);
    float* h_x1 = (float*)malloc(bytes_v);
    float* h_x2 = (float*)malloc(bytes_v);

    // Initialize with random floats
    srand(42);
    for (int i = 0; i < n * n; ++i)
        h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int j = 0; j < n; ++j) {
        h_y1[j] = (float)(rand() % 100) / 10.0f;
        h_y2[j] = (float)(rand() % 100) / 10.0f;
        h_x1[j] = 0.0f;
        h_x2[j] = 0.0f;
    }

    // Device allocations
    float *d_A, *d_y1, *d_y2, *d_x1, *d_x2;
    HIP_CHECK(hipMalloc(&d_A,  bytes_A));
    HIP_CHECK(hipMalloc(&d_y1, bytes_v));
    HIP_CHECK(hipMalloc(&d_y2, bytes_v));
    HIP_CHECK(hipMalloc(&d_x1, bytes_v));
    HIP_CHECK(hipMalloc(&d_x2, bytes_v));

    HIP_CHECK(hipMemcpy(d_A,  h_A,  bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y1, h_y1, bytes_v, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y2, h_y2, bytes_v, hipMemcpyHostToDevice));

    // Initialize output vectors to zero on device
    HIP_CHECK(hipMemset(d_x1, 0, bytes_v));
    HIP_CHECK(hipMemset(d_x2, 0, bytes_v));

    int block_size = 256;
    int grid       = (n + block_size - 1) / block_size;

    char params[128];
    snprintf(params, sizeof(params),
             "\"n\":%d", n);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2;

    // One full MVT pass (run exactly once): both kernels, each timed
    // separately. The kernels rewrite x1/x2 as pure functions of the
    // constant A/y1/y2 inputs, so every execution does identical work — no
    // reset needed.
    auto run_mvt_pass = [&]() {
        bench_lap_reset(&acc_k1);
        bench_lap_reset(&acc_k2);

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
            mvt_kernel1<<<grid, block_size>>>(d_A, d_y1, d_x1, n);
        }));
        bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
            mvt_kernel2<<<grid, block_size>>>(d_A, d_y2, d_x2, n);
        }));
    };

    run_mvt_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("mvt_kernel1", 0, &acc_k1, params);
    bench_emit_iter("mvt_kernel2", 0, &acc_k2, params);
    total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms;
    fprintf(stderr, "mvt_kernel1 gpu = %.3f ms, mvt_kernel2 gpu = %.3f ms\n",
            acc_k1.gpu_ms, acc_k2.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over 1 pass\n",
            total_gpu_ms);

    // Copy results back for correctness check
    HIP_CHECK(hipMemcpy(h_x1, d_x1, bytes_v, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_x2, d_x2, bytes_v, hipMemcpyDeviceToHost));

    // Correctness check against CPU reference (small subset)
    int check_n = (n < 8) ? n : 8;
    std::vector<float> x1_ref(n), x2_ref(n);
    cpu_mvt(h_A, h_y1, h_y2, x1_ref.data(), x2_ref.data(), n);

    bool pass = true;
    for (int i = 0; i < check_n; ++i) {
        float rel1 = fabsf(h_x1[i] - x1_ref[i]) / (fabsf(x1_ref[i]) + 1e-6f);
        if (rel1 > 1e-3f) {
            fprintf(stderr, "MISMATCH x1[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, h_x1[i], x1_ref[i], rel1);
            pass = false;
        }
        float rel2 = fabsf(h_x2[i] - x2_ref[i]) / (fabsf(x2_ref[i]) + 1e-6f);
        if (rel2 > 1e-3f) {
            fprintf(stderr, "MISMATCH x2[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, h_x2[i], x2_ref[i], rel2);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (first %d elements): %s\n\n",
            check_n, pass ? "PASS" : "FAIL");

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_y1));
    HIP_CHECK(hipFree(d_y2));
    HIP_CHECK(hipFree(d_x1));
    HIP_CHECK(hipFree(d_x2));
    free(h_A);
    free(h_y1);
    free(h_y2);
    free(h_x1);
    free(h_x2);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
