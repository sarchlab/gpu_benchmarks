// parboil_sgemm.hip — Parboil SGEMM benchmark (HIP, self-contained)
//
// Tiled single-precision GEMM: C = alpha*A*B + beta*C for NxN matrices.
//
// Derived from the Parboil benchmark suite SGEMM workload.
//
// Compiles with hipcc (AMD ROCm) or nvcc (CUDA via HIP compat layer).
//
// Usage:
//   ./parboil_sgemm [--size N]
//
//   --size N         Matrix dimension N (NxN)  (default: 1024)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_precision
//   BENCH_PARAM_warmup   (warmup launches, default 3)
//   BENCH_PARAM_iteration    (timed launches, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// carrying the GPU event time (time_ms) and host wall-clock time
// (host_time_ms), then a summary whose total_time_ms is the SUM of all
// iterations' GPU event times. GFLOPS (2*N^3 / time) is derived downstream
// from time_ms and params.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
// CPU reference for correctness check
// ---------------------------------------------------------------------------

static void sgemm_cpu(const float* A, const float* B, float* C,
                      int N, float alpha, float beta) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) {
                s += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * s + beta * C[i * N + j];
        }
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
    int block_size = 256;
    const char* precision = "float";

    // Command-line args as fallback
    N     = parseIntParam(argc, argv, "--size", N);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) {
        // Validate against the known enum set so the emitted params string is
        // always a canonical literal, never raw env text.
        if      (strcmp(env_val, "float")  == 0) precision = "float";
        else if (strcmp(env_val, "double") == 0) precision = "double";
        else if (strcmp(env_val, "half")   == 0) precision = "half";
        else {
            fprintf(stderr, "Invalid BENCH_PARAM_precision (expected float, "
                            "double or half)\n");
            return 1;
        }
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    float alpha = 1.5f;
    float beta  = 1.2f;

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
    float* h_A   = (float*)malloc(bytes);
    float* h_B   = (float*)malloc(bytes);
    float* h_C   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

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
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"precision\":\"%s\"",
             N, block_size, precision);

    // Warmup + timed iterations, each recorded individually. The kernel
    // rewrites C in place (beta*C term); float FMA timing is value-
    // independent, so iterating without re-uploading C does not change the
    // measured time and keeps the launch loop free of setup copies.
    double total_gpu_ms =
        bench_run_loop("sgemm_kernel", num_warmup, num_iters, params, [&]() {
            sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, N, alpha, beta);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    // Correctness check (only for small N to keep runtime short)
    if (N <= 256) {
        // The benchmark loop applied the in-place update repeatedly; redo a
        // single application from the pristine C for verification.
        HIP_CHECK(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice));
        sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, N, alpha, beta);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));
        sgemm_cpu(h_A, h_B, h_ref, N, alpha, beta);

        int errors = 0;
        for (int i = 0; i < N * N; ++i) {
            float diff = fabsf(h_C[i] - h_ref[i]);
            if (diff > 1e-3f * fabsf(h_ref[i]) + 1e-4f) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at %d: got %f, expected %f\n",
                            i, h_C[i], h_ref[i]);
                }
                errors++;
            }
        }
        if (errors > 0)
            fprintf(stderr, "FAIL: %d errors\n", errors);
        else
            fprintf(stderr, "PASS\n");
    } else {
        fprintf(stderr, "Skipping verification for large size\n");
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
