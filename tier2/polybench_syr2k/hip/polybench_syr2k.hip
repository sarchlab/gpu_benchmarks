// polybench_syr2k.hip — PolyBench SYR2K benchmark (HIP, self-contained)
//
// Rank-2k update: C = alpha*A*B^T + alpha*B*A^T + beta*C
// A and B are NxM matrices, C is NxN. Note: C is initialized with random
// (non-symmetric) values and the kernel updates the FULL matrix, so unlike
// BLAS SYR2K the stored C is not symmetric; the alpha*(A*B^T + B*A^T)
// update term itself is symmetric.
// Modeled on the PolyBench/GPU SYR2K workload (tiled reimplementation).
//
// Default: N=1024, M=1024, alpha=1.5, beta=1.2
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./polybench_syr2k [--size N] [--inner_size M] [--alpha A] [--beta B]
//
// Environment variables (override argv):
//   BENCH_PARAM_matrix_size, BENCH_PARAM_inner_size,
//   BENCH_PARAM_alpha, BENCH_PARAM_beta
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// launch, carrying the GPU event time and host wall-clock time, then a
// summary whose total_time_ms equals that GPU event time. GFLOPS
// (4*N^2*M / time) is derived downstream from time_ms and params.
// Output (stderr): device info, timing

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cpu_reference.h"

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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

static float parseFloatParam(int argc, char** argv, const char* name, float defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            float v = (float)atof(argv[i + 1]);
            return v;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// SYR2K kernel — tiled shared-memory implementation
// C = alpha*A*B^T + alpha*B*A^T + beta*C
// A is [N x M], B is [N x M], C is [N x N]
//
// C[row][col] = alpha * sum_k(A[row][k]*B[col][k])
//             + alpha * sum_k(B[row][k]*A[col][k])
//             + beta  * C[row][col]
// ---------------------------------------------------------------------------
#define TILE_SIZE 16

__global__ void syr2k_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N, int M,
                              float alpha, float beta) {
    __shared__ float sA_row[TILE_SIZE][TILE_SIZE];
    __shared__ float sB_row[TILE_SIZE][TILE_SIZE];
    __shared__ float sA_col[TILE_SIZE][TILE_SIZE];
    __shared__ float sB_col[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int k = t * TILE_SIZE + threadIdx.x;

        // Load A and B rows for the row-block
        sA_row[threadIdx.y][threadIdx.x] = (row < N && k < M) ? A[row * M + k] : 0.0f;
        sB_row[threadIdx.y][threadIdx.x] = (row < N && k < M) ? B[row * M + k] : 0.0f;

        // Load A and B rows for the col-block
        int col_row = blockIdx.x * TILE_SIZE + threadIdx.y;
        int k2 = t * TILE_SIZE + threadIdx.x;
        sA_col[threadIdx.y][threadIdx.x] = (col_row < N && k2 < M) ? A[col_row * M + k2] : 0.0f;
        sB_col[threadIdx.y][threadIdx.x] = (col_row < N && k2 < M) ? B[col_row * M + k2] : 0.0f;

        __syncthreads();

        for (int kk = 0; kk < TILE_SIZE; ++kk) {
            // A*B^T: A[row][k]*B[col][k]  +  B*A^T: B[row][k]*A[col][k]
            sum += sA_row[threadIdx.y][kk] * sB_col[threadIdx.x][kk]
                 + sB_row[threadIdx.y][kk] * sA_col[threadIdx.x][kk];
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
    int M     = parseIntParam(argc, argv, "--inner_size", 1024);
    float alpha = parseFloatParam(argc, argv, "--alpha", 1.5f);
    float beta  = parseFloatParam(argc, argv, "--beta", 1.2f);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_inner_size");
    if (env_val) M = atoi(env_val);
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha = (float)atof(env_val);
    env_val = getenv("BENCH_PARAM_beta");
    if (env_val) beta = (float)atof(env_val);
    size_t bytes_AB = (size_t)N * M * sizeof(float);
    size_t bytes_C  = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "SYR2K: N=%d, M=%d  |  alpha=%.1f  beta=%.1f\n\n",
            N, M, alpha, beta);

    // Host allocations
    float* h_A = (float*)malloc(bytes_AB);
    float* h_B = (float*)malloc(bytes_AB);
    float* h_C = (float*)malloc(bytes_C);

    // PolyBench-style initialization
    srand(42);
    for (int i = 0; i < N * M; ++i) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < N * N; ++i) {
        h_C[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes_AB));
    HIP_CHECK(hipMalloc(&d_B, bytes_AB));
    HIP_CHECK(hipMalloc(&d_C, bytes_C));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_AB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes_AB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, bytes_C, hipMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    char params[256];
    snprintf(params, sizeof(params),
             "\"matrix_size\":%d,\"inner_size\":%d,"
             "\"alpha\":%.2f,\"beta\":%.2f,\"verification\":\"full_cpu_v1\"",
             N, M, alpha, beta);

    // Single timed launch (optionally preceded by the vector-add warmer).
    // The kernel rewrites C in place (beta*C term), but it runs only once
    // per process on the freshly uploaded C, so every execution performs
    // identical work.
    double total_gpu_ms =
        bench_run_once("syr2k_kernel", params, [&]() {
            syr2k_kernel<<<grid, block>>>(d_A, d_B, d_C, N, M, alpha, beta);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify the complete measured result after timing, without another GPU pass.
    const auto expected = cpu_syr2k(h_A, h_B, h_C, N, M, alpha, beta);
    HIP_CHECK(hipMemcpy(h_C, d_C, bytes_C, hipMemcpyDeviceToHost));
    const size_t elements = (size_t)N * N;
    size_t mismatches = 0;
    for (size_t i = 0; i < elements; ++i) {
        if (!syr2k_value_matches(h_C[i], expected[i])) {
            if (mismatches < 10)
                fprintf(stderr, "MISMATCH C[%zu]: GPU=%.9g CPU=%.17g\n",
                        i, h_C[i], expected[i]);
            ++mismatches;
        }
    }
    fprintf(stderr, "SYR2K verification: %s (%zu elements, %zu mismatches)\n",
            mismatches ? "FAIL" : "PASS", elements, mismatches);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
