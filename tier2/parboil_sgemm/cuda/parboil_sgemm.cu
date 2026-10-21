// parboil_sgemm.cu — Parboil SGEMM benchmark (CUDA, self-contained)
//
// Tiled single-precision GEMM: C = alpha*A*B + beta*C for NxN matrices.
//
// Derived from the Parboil benchmark suite SGEMM workload.
//
// Native CUDA port of ../hip/parboil_sgemm.hip; compiles with nvcc.
//
// Usage:
//   ./parboil_sgemm [--size N]
//
//   --size N         Matrix dimension N (NxN)  (default: 1024)
//
// Environment variables (override argv):
//   BENCH_PARAM_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// sgemm_kernel launch carrying the GPU event time (time_ms) and host
// wall-clock time (host_time_ms), then a summary whose total_time_ms is the
// sum of all recorded GPU event times. GFLOPS (2*N^3 / time) is derived
// downstream from time_ms and params.

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "cpu_reference.h"

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

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
// Tiled SGEMM kernel (16x16 shared-memory tiles)
// ---------------------------------------------------------------------------

#define TILE 16

__global__ void sgemm_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N, float alpha, float beta)
{
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        tA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        tB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Default values from params.json
    int N     = 1024;

    // Command-line args as fallback
    N     = parseIntParam(argc, argv, "--size", N);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    float alpha = 1.5f;
    float beta  = 1.2f;

    size_t bytes = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d  |  alpha=%.1f  beta=%.1f\n\n",
            N, N, alpha, beta);

    // Host allocations
    float* h_A   = (float*)malloc(bytes);
    float* h_B   = (float*)malloc(bytes);
    float* h_C   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "FAIL: complete host matrices unavailable\n");
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return EXIT_FAILURE;
    }

    // Initialize inputs
    for (int i = 0; i < N * N; ++i)
        h_A[i] = (float)(i % 1000) * 0.001f;
    for (int i = 0; i < N * N; ++i)
        h_B[i] = (float)((i + 37) % 1000) * 0.001f;
    for (int i = 0; i < N * N; ++i) {
        h_C[i]   = (float)(i % 100) * 0.01f;
        h_ref[i] = h_C[i];
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"verification\":\"full_cpu_v1\"",
             N);

    // Single timed launch (bench_run_once): the kernel rewrites C in place
    // (beta*C term), but it runs exactly once per process, so no reset is
    // needed before the measurement.
    double total_gpu_ms =
        bench_run_once("sgemm_kernel", params, [&]() {
            sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, N, alpha, beta);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify every measured output after timing; no skipped sizes or GPU rerun.
    bool verify_ok = false;
    try {
        // h_ref retains pristine C; d_C holds the one measured update.
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        gemm_cpu(h_A, h_B, h_ref, N, N, N, alpha, beta);

        const size_t count = (size_t)N * N;
        size_t errors = 0;
        for (size_t i = 0; i < count; ++i) {
            float diff = fabsf(h_C[i] - h_ref[i]);
            if (!std::isfinite(h_C[i]) || !std::isfinite(h_ref[i]) ||
                diff > 1e-3f * fabsf(h_ref[i]) + 1e-4f) {
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

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return verify_ok ? 0 : 1;
}
