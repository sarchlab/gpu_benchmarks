// shoc_gemm.hip — SHOC GEMM benchmark (HIP, self-contained)
//
// Dense matrix multiply: C = alpha*A*B + beta*C for NxN square matrices of floats.
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback:
//   BENCH_PARAM_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// launch carrying the GPU event time (time_ms) and host wall-clock time
// (host_time_ms), then a summary whose total_time_ms equals that record's
// GPU event time. GFLOPS (2*N^3 / time) is derived downstream from time_ms
// and params.
// Output (stderr): Human-readable diagnostics

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "cpu_reference.h"

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
// GEMM kernel
// ---------------------------------------------------------------------------
#define TILE_SIZE 16

__global__ void gemm_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K,
                             float alpha, float beta) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int size = 512;

    // Command-line fallback
    size = parseIntParam(argc, argv, "--size", size);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) size = atoi(env_val);

    int M = size, N = size, K = size;
    float alpha = 1.0f, beta = 0.0f;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d\n\n", size, size);

    // Host allocations
    float* h_A   = (float*)malloc(bytesA);
    float* h_B   = (float*)malloc(bytesB);
    float* h_C   = (float*)malloc(bytesC);
    float* h_ref = (float*)malloc(bytesC);
    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "FAIL: complete host matrices unavailable\n");
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * K; ++i)
        h_A[i] = (float)(i % 1000) * 0.001f;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (float)((i + 37) % 1000) * 0.001f;
    for (int i = 0; i < M * N; ++i) {
        h_C[i]   = 0.0f;
        h_ref[i] = 0.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytesA));
    HIP_CHECK(hipMalloc(&d_B, bytesB));
    HIP_CHECK(hipMalloc(&d_C, bytesC));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, bytesC, hipMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    char params[160];
    snprintf(params, sizeof(params), "\"size\":%d,\"verification\":\"full_cpu_v1\"", size);

    // Single timed launch (bench_run_once: optional cold-cache warmer, then
    // one bracketed launch). With beta = 0 the kernel fully overwrites C
    // (the beta*C term is zero), so every execution performs identical work
    // — no reset needed.
    double total_gpu_ms =
        bench_run_once("gemm_kernel", params, [&]() {
            gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify every measured output after timing; no skipped sizes or GPU rerun.
    bool verify_ok = false;
    try {
        HIP_CHECK(hipMemcpy(h_C, d_C, bytesC, hipMemcpyDeviceToHost));
        gemm_cpu(h_A, h_B, h_ref, M, N, K, alpha, beta);

        const size_t count = (size_t)M * N;
        size_t errors = 0;
        for (size_t i = 0; i < count; ++i) {
            float diff = fabsf(h_C[i] - h_ref[i]);
            if (!std::isfinite(h_C[i]) || !std::isfinite(h_ref[i]) ||
                diff > 1e-3f * fabsf(h_ref[i]) + 1e-5f) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at %zu: got %f, expected %f\n",
                            i, h_C[i], h_ref[i]);
                }
                errors++;
            }
        }
        verify_ok = errors == 0;
        fprintf(stderr, "%s: %zu mismatches in %zu values\n",
                verify_ok ? "PASS" : "FAIL", errors, count);
    } catch (const std::exception& error) {
        fprintf(stderr, "FAIL: complete CPU reference unavailable: %s\n", error.what());
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
