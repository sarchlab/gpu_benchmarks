// polybench_2mm.hip — PolyBench 2MM benchmark: Two Matrix Multiplications (HIP)
//
// Computes D = alpha * A * B + beta * D, then E = alpha * C * D + beta * E
// for NxN square matrices. Tiled shared-memory implementation.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_alpha, BENCH_PARAM_beta
//
// Output (stdout): the 2MM pass runs exactly once; one {"type":"kernel",...}
// JSON line per kernel (mm_kernel_1, mm_kernel_2) carrying its GPU event time
// (time_ms) and host wall-clock time (host_time_ms), then a summary whose
// total_time_ms is the SUM of the two GPU event times. GFLOPS
// (4*N^3 / time) is derived downstream from time_ms and params.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
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
// Tiled GEMM kernel: Out = alpha * P * Q + beta * Out  (NxN matrices)
// ---------------------------------------------------------------------------

#define TILE_SIZE 16

__global__ void mm_kernel(const float* __restrict__ P,
                          const float* __restrict__ Q,
                          float* __restrict__ Out,
                          int N, float alpha, float beta) {
    __shared__ float sP[TILE_SIZE][TILE_SIZE];
    __shared__ float sQ[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int pCol = t * TILE_SIZE + threadIdx.x;
        int qRow = t * TILE_SIZE + threadIdx.y;

        sP[threadIdx.y][threadIdx.x] = (row < N && pCol < N) ? P[row * N + pCol] : 0.0f;
        sQ[threadIdx.y][threadIdx.x] = (qRow < N && col < N) ? Q[qRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sP[threadIdx.y][k] * sQ[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        Out[row * N + col] = alpha * sum + beta * Out[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N          = 1024;
    float alpha    = 1.5f;
    float beta     = 1.2f;

    N          = parseIntParam(argc, argv, "--size", N);
    alpha      = parseFloatParam(argc, argv, "--alpha", alpha);
    beta       = parseFloatParam(argc, argv, "--beta", beta);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha = (float)atof(env_val);
    env_val = getenv("BENCH_PARAM_beta");
    if (env_val) beta = (float)atof(env_val);
    size_t bytes = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "2MM: D=alpha*A*B+beta*D, E=alpha*C*D+beta*E  |  N=%d  |  "
            "alpha=%.1f  beta=%.1f\n\n",
            N, alpha, beta);

    // Host allocations
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_D = (float*)malloc(bytes);
    float* h_E = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
        h_C[i] = (float)(rand() % 100) / 10.0f;
        h_D[i] = (float)(rand() % 100) / 10.0f;
        h_E[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C, *d_D, *d_E;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));
    HIP_CHECK(hipMalloc(&d_D, bytes));
    HIP_CHECK(hipMalloc(&d_E, bytes));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    char params[192];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"alpha\":%.2f,\"beta\":%.2f",
             N, alpha, beta);

    // Per-kernel accumulators: one launch of each mm_kernel per pass.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_mm1, acc_mm2;

    // One full 2MM pass. D and E are rewritten in place (beta*Out term), so
    // the pristine inputs are re-uploaded at the start of every pass
    // (untimed) — the same reset the original code performed.
    auto run_2mm_pass = [&]() {
        bench_lap_reset(&acc_mm1);
        bench_lap_reset(&acc_mm2);

        HIP_CHECK(hipMemcpy(d_D, h_D, bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_E, h_E, bytes, hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        // mm_kernel #1: D = alpha*A*B + beta*D
        bench_lap_add(&acc_mm1, bench_time_op(&timer, [&]() {
            mm_kernel<<<grid, block>>>(d_A, d_B, d_D, N, alpha, beta);
        }));

        // mm_kernel #2: E = alpha*C*D + beta*E
        bench_lap_add(&acc_mm2, bench_time_op(&timer, [&]() {
            mm_kernel<<<grid, block>>>(d_C, d_D, d_E, N, alpha, beta);
        }));
    };

    run_2mm_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("mm_kernel_1", 0, &acc_mm1, params);
    bench_emit_iter("mm_kernel_2", 0, &acc_mm2, params);
    total_gpu_ms += acc_mm1.gpu_ms + acc_mm2.gpu_ms;
    fprintf(stderr, "mm1 gpu = %.3f ms, mm2 gpu = %.3f ms\n",
            acc_mm1.gpu_ms, acc_mm2.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_E));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);

    return 0;
}
