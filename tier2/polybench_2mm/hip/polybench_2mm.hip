// polybench_2mm.hip — PolyBench 2MM benchmark: Two Matrix Multiplications (HIP)
//
// Computes tmp = alpha * A * B, then D = tmp * C + beta * D
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

#include "../../../common_hip/bench_timing.h"
#include "cpu_reference.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

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

template<bool Accumulate>
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
        // The temporary has no initial value: stage 1 must never read it.
        float value = alpha * sum;
        if (Accumulate) value += beta * Out[row * N + col];
        Out[row * N + col] = value;
    }
}

// Both main and the GPU reference test use this complete production pass.
struct TwoMMLaps {
    bench_lap_t first;
    bench_lap_t second;
};

static TwoMMLaps run_2mm_pass(const float *A, const float *B, const float *C,
                            float *D, float *tmp, int N, float alpha, float beta) {
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                    (N + TILE_SIZE - 1) / TILE_SIZE);
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_pass_warm();
    TwoMMLaps laps;
    // The upstream dependency is (A B) C; no initial temporary contribution.
    laps.first = bench_time_op(&timer, [&]() {
        mm_kernel<false><<<grid, block>>>(A, B, tmp, N, alpha, 0.0f);
    });
    laps.second = bench_time_op(&timer, [&]() {
        mm_kernel<true><<<grid, block>>>(tmp, C, D, N, 1.0f, beta);
    });
    bench_timer_destroy(&timer);
    return laps;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int N = parseIntParam(argc, argv, "--size", 1024);
    float alpha = parseFloatParam(argc, argv, "--alpha", 1.5f);
    float beta = parseFloatParam(argc, argv, "--beta", 1.2f);
    const char *env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha = (float)atof(env_val);
    env_val = getenv("BENCH_PARAM_beta");
    if (env_val) beta = (float)atof(env_val);
    // Matrix indices in the tiled kernel are signed ints.
    if (N <= 0 || N > std::numeric_limits<int>::max() / N ||
        static_cast<size_t>(N) > std::numeric_limits<size_t>::max() / sizeof(float) / N ||
        !std::isfinite(alpha) || !std::isfinite(beta)) {
        fprintf(stderr, "Invalid matrix extent or non-finite coefficient\n");
        return EXIT_FAILURE;
    }
    const size_t elements = static_cast<size_t>(N) * N;
    const size_t bytes = elements * sizeof(float);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "2MM: tmp=alpha*A*B, D=tmp*C+beta*D | N=%d | alpha=%.2f beta=%.2f\n",
            N, alpha, beta);

    // Four inputs; the GPU temporary is written completely by the first stage.
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Host matrix allocation failed\n");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    srand(42);
    for (size_t i = 0; i < elements; ++i) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
        h_C[i] = (float)(rand() % 100) / 10.0f;
        h_D[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_A, *d_B, *d_C, *d_D, *d_tmp;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));
    HIP_CHECK(hipMalloc(&d_D, bytes));
    HIP_CHECK(hipMalloc(&d_tmp, bytes));
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_D, h_D, bytes, hipMemcpyHostToDevice));

    TwoMMLaps laps = run_2mm_pass(d_A, d_B, d_C, d_D, d_tmp, N, alpha, beta);
    char params[192];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"alpha\":%.2f,\"beta\":%.2f,\"verification\":\"full_cpu_v1\"", N, alpha, beta);
    bench_emit_iter("mm_kernel_1", 0, &laps.first, params);
    bench_emit_iter("mm_kernel_2", 0, &laps.second, params);
    const double total_gpu_ms = laps.first.gpu_ms + laps.second.gpu_ms;
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "mm1 gpu = %.3f ms, mm2 gpu = %.3f ms\nTotal GPU time: %.3f ms\n",
            laps.first.gpu_ms, laps.second.gpu_ms, total_gpu_ms);

    // Verify both measured products after timing, with no extra GPU pass.
    // Consume all original inputs before reusing h_A and h_D for the outputs.
    const CpuTwoMM reference = cpu_2mm(h_A, h_B, h_C, h_D, N, alpha, beta);
    HIP_CHECK(hipMemcpy(h_A, d_tmp, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_D, d_D, bytes, hipMemcpyDeviceToHost));
    size_t errors = 0;
    for (size_t i = 0; i < elements; ++i)
        for (int stage = 0; stage < 2; ++stage) {
            const float actual = stage ? h_D[i] : h_A[i];
            const double expected = stage ? reference.output[i] : reference.tmp[i];
            if (!two_mm_value_matches(actual, expected)) {
                if (errors < 10)
                    fprintf(stderr, "MISMATCH %s[%zu]: got %.9g expected %.12g\n",
                            stage ? "D" : "tmp", i, actual, expected);
                ++errors;
            }
        }
    fprintf(stderr, "2MM verification: %s (%zu elements per matrix, %zu mismatches)\n",
            errors ? "FAIL" : "PASS", elements, errors);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_tmp));
    free(h_A); free(h_B); free(h_C); free(h_D);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
