// polybench_atax.hip — PolyBench ATAX benchmark (HIP, self-contained)
//
// Computes y = A^T * (A * x) where A is NX×NY, x is NY-vector, y is NY-vector.
// Derived from the PolyBench/GPU benchmark suite.
//
// Two kernels:
//   atax_kernel1: tmp[i] = sum_j A[i,j] * x[j]   (tmp = A * x)
//   atax_kernel2: y[j]   = sum_i A[i,j] * tmp[i]  (y = A^T * tmp)
//
// Default: NX = NY = 4096
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./polybench_atax [--nx NX] [--ny NY]
//
//   --nx NX          Number of rows    (default: 4096)
//   --ny NY          Number of columns (default: 4096)
//
// Environment variables (override argv):
//   BENCH_PARAM_nx, BENCH_PARAM_ny, BENCH_PARAM_block_size
//
// Output (stdout): the ATAX pass runs exactly once; one {"type":"kernel",...}
// JSON line per kernel (atax_kernel1, atax_kernel2) carrying that kernel's
// GPU event time and host time, then a summary whose total_time_ms is the
// SUM of the two GPU event times. Bandwidth (A read twice:
// 2*NX*NY*4 bytes / time) is derived downstream from time_ms and params.
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
// Kernel 1: tmp = A * x
// Each thread computes one row: tmp[i] = sum_j A[i,j] * x[j]
// ---------------------------------------------------------------------------
__global__ void atax_kernel1(const float* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ tmp,
                              int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nx) {
        float sum = 0.0f;
        for (int j = 0; j < ny; j++)
            sum += A[i * ny + j] * x[j];
        tmp[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: y = A^T * tmp
// Each thread computes one column: y[j] = sum_i A[i,j] * tmp[i]
// ---------------------------------------------------------------------------
__global__ void atax_kernel2(const float* __restrict__ A,
                              const float* __restrict__ tmp,
                              float* __restrict__ y,
                              int nx, int ny) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < ny) {
        float sum = 0.0f;
        for (int i = 0; i < nx; i++)
            sum += A[i * ny + j] * tmp[i];
        y[j] = sum;
    }
}

// ---------------------------------------------------------------------------
// CPU reference for correctness check
// ---------------------------------------------------------------------------
static void cpu_atax(const float* A, const float* x, float* y_ref, int nx, int ny) {
    // tmp = A * x
    std::vector<float> tmp(nx, 0.0f);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            tmp[i] += A[i * ny + j] * x[j];
    // y = A^T * tmp
    for (int j = 0; j < ny; j++) {
        y_ref[j] = 0.0f;
        for (int i = 0; i < nx; i++)
            y_ref[j] += A[i * ny + j] * tmp[i];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int nx    = parseIntParam(argc, argv, "--nx", 4096);
    int ny    = parseIntParam(argc, argv, "--ny", 4096);
    int block_size = 256;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_nx");
    if (env_val) nx = atoi(env_val);
    env_val = getenv("BENCH_PARAM_ny");
    if (env_val) ny = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    size_t bytes_A   = (size_t)nx * ny * sizeof(float);
    size_t bytes_x   = (size_t)ny * sizeof(float);
    size_t bytes_tmp = (size_t)nx * sizeof(float);
    size_t bytes_y   = (size_t)ny * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d\n\n", nx, ny);

    // Host allocations
    float* h_A = (float*)malloc(bytes_A);
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_y);

    // Initialize (random floats)
    srand(42);
    for (int i = 0; i < nx * ny; ++i)
        h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int j = 0; j < ny; ++j)
        h_x[j] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_x, *d_tmp, *d_y;
    HIP_CHECK(hipMalloc(&d_A,   bytes_A));
    HIP_CHECK(hipMalloc(&d_x,   bytes_x));
    HIP_CHECK(hipMalloc(&d_tmp, bytes_tmp));
    HIP_CHECK(hipMalloc(&d_y,   bytes_y));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, bytes_x, hipMemcpyHostToDevice));

    int grid1 = (nx + block_size - 1) / block_size;
    int grid2 = (ny + block_size - 1) / block_size;

    char params[128];
    snprintf(params, sizeof(params),
             "\"nx\":%d,\"ny\":%d,\"block_size\":%d",
             nx, ny, block_size);

    // One full ATAX pass (run exactly once): kernel1 writes d_tmp, kernel2
    // consumes it and writes d_y. A and x are never modified and both
    // outputs are fully overwritten, so every execution performs identical
    // work — no state reset is needed.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2;

    auto run_atax_pass = [&]() {
        bench_lap_reset(&acc_k1);
        bench_lap_reset(&acc_k2);

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
            atax_kernel1<<<grid1, block_size>>>(d_A, d_x, d_tmp, nx, ny);
        }));
        bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
            atax_kernel2<<<grid2, block_size>>>(d_A, d_tmp, d_y, nx, ny);
        }));
    };

    run_atax_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("atax_kernel1", 0, &acc_k1, params);
    bench_emit_iter("atax_kernel2", 0, &acc_k2, params);
    total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms;
    fprintf(stderr, "k1 gpu = %.3f ms, k2 gpu = %.3f ms\n",
            acc_k1.gpu_ms, acc_k2.gpu_ms);

    bench_timer_destroy(&timer);

    // Copy result back for correctness check
    HIP_CHECK(hipMemcpy(h_y, d_y, bytes_y, hipMemcpyDeviceToHost));

    // Correctness check against CPU reference (small subset)
    int check_n = (ny < 8) ? ny : 8;
    std::vector<float> y_ref(ny);
    cpu_atax(h_A, h_x, y_ref.data(), nx, ny);
    bool pass = true;
    for (int j = 0; j < check_n; ++j) {
        float rel = fabsf(h_y[j] - y_ref[j]) / (fabsf(y_ref[j]) + 1e-6f);
        if (!std::isfinite(h_y[j]) || !std::isfinite(y_ref[j]) || !std::isfinite(rel) ||
            rel > 1e-3f) {
            fprintf(stderr, "MISMATCH at y[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    j, h_y[j], y_ref[j], rel);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (first %d elements): %s\n\n",
            check_n, pass ? "PASS" : "FAIL");

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_tmp));
    HIP_CHECK(hipFree(d_y));
    free(h_A);
    free(h_x);
    free(h_y);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
