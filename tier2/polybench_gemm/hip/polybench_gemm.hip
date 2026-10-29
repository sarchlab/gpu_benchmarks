// polybench_gemm.hip — PolyBench GEMM benchmark (HIP, self-contained)
//
// Dense matrix multiply: C = alpha*A*B + beta*C for NxN square matrices.
// Derived from the PolyBench/GPU benchmark suite.
//
// alpha = 1.5, beta = 1.2, default N = 1024
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./polybench_gemm [--size N]
//
//   --size N         Matrix dimension N (N×N)  (default: 1024)
//
// Environment variables (override argv):
//   BENCH_PARAM_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// launch, carrying the GPU event time and host wall-clock time, then a
// summary whose total_time_ms equals that GPU event time. GFLOPS
// (2*N^3 / time) is derived downstream from time_ms and params.

#include "../../../common_hip/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// PolyBench GEMM kernel — tiled 16×16 shared-memory GEMM
// C[row][col] = alpha * sum_k(A[row][k] * B[k][col]) + beta * C[row][col]
// ---------------------------------------------------------------------------
#define TILE_SIZE 16

__global__ void polybench_gemm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N,
                                       float alpha, float beta) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
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
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N     = parseIntParam(argc, argv, "--size", 1024);
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    const float alpha = 1.5f;
    const float beta  = 1.2f;

    size_t bytes = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d  |  alpha=%.1f  beta=%.1f\n\n",
            N, N, alpha, beta);

    // Host allocations
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize (PolyBench-style initialization)
    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
        h_C[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"verification\":\"full_cpu_v1\"", N);

    // Single timed launch (optionally preceded by the vector-add warmer).
    // The kernel rewrites C in place (beta*C term), but it runs only once
    // per process on the freshly uploaded C, so every execution performs
    // identical work.
    double total_gpu_ms =
        bench_run_once("polybench_gemm_kernel", params, [&]() {
            polybench_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, N, alpha, beta);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Check the complete measured C after timing, without another GPU pass.
    const auto reference = cpu_gemm(h_A, h_B, h_C, N, N, N, alpha, beta);
    // The reference has consumed initial C, so its buffer can hold the result.
    HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!gemm_value_matches(h_C[i], reference[i])) {
            if (mismatches < 10)
                fprintf(stderr, "MISMATCH C[%zu]: GPU=%.9g CPU=%.17g\n",
                        i, h_C[i], reference[i]);
            ++mismatches;
        }
    }
    fprintf(stderr, "GEMM verification: %s (%zu values, %zu mismatches)\n",
            mismatches ? "FAIL" : "PASS", reference.size(), mismatches);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
