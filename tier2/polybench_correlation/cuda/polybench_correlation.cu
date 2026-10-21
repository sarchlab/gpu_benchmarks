// polybench_correlation.cu — PolyBench Correlation benchmark (CUDA, self-contained)
//
// Compute correlation matrix from a data matrix (M samples x N features).
// Steps: mean, stddev, normalize, correlation (matmul-like).
// Derived from the PolyBench/GPU benchmark suite.
//
// Native CUDA port of ../hip/polybench_correlation.hip; compiles with nvcc.
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size
//
// Output (stdout): the correlation pass runs exactly once; one
// {"type":"kernel",...} JSON line per kernel (mean_kernel, stddev_kernel,
// normalize_kernel, correlation_kernel) carrying that kernel's GPU event time
// and host time, then a summary whose total_time_ms is the SUM of the four
// GPU event times. Derived metrics are computed downstream from time_ms and
// params.
// Output (stderr): device info, timing details

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Kernel 1: Compute column means
// ---------------------------------------------------------------------------
__global__ void mean_kernel(const float* __restrict__ data,
                            float* __restrict__ mean,
                            int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < M; ++i) {
        sum += data[i * N + j];
    }
    mean[j] = sum / (float)M;
}

// ---------------------------------------------------------------------------
// Kernel 2: Compute column standard deviations
// ---------------------------------------------------------------------------
__global__ void stddev_kernel(const float* __restrict__ data,
                              const float* __restrict__ mean,
                              float* __restrict__ stddev,
                              int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    float m = mean[j];
    float sum = 0.0f;
    for (int i = 0; i < M; ++i) {
        float diff = data[i * N + j] - m;
        sum += diff * diff;
    }
    float s = sqrtf(sum / (float)M);
    stddev[j] = (s < 1e-12f) ? 1.0f : s;
}

// ---------------------------------------------------------------------------
// Kernel 3: Normalize data
// ---------------------------------------------------------------------------
__global__ void normalize_kernel(float* __restrict__ data,
                                 const float* __restrict__ mean,
                                 const float* __restrict__ stddev,
                                 int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int j = idx % N;
    float sqrt_m = sqrtf((float)M);
    data[idx] = (data[idx] - mean[j]) / (sqrt_m * stddev[j]);
}

// ---------------------------------------------------------------------------
// Kernel 4: Compute correlation matrix (tiled)
// ---------------------------------------------------------------------------
#define TILE_SIZE 16

__global__ void correlation_kernel(const float* __restrict__ data,
                                   float* __restrict__ corr,
                                   int M, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int k_a = t * TILE_SIZE + threadIdx.x;
        int k_b = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && k_a < M) ? data[k_a * N + row] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (k_b < M && col < N) ? data[k_b * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        if (row == col)
            corr[row * N + col] = 1.0f;
        else
            corr[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N          = parseIntParam(argc, argv, "--size", 1024);
    int block_size = parseIntParam(argc, argv, "--block_size", 256);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int M = N;

    size_t data_bytes = (size_t)M * N * sizeof(float);
    size_t vec_bytes  = (size_t)N * sizeof(float);
    size_t corr_bytes = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Data matrix: %dx%d\n\n", M, N);

    // Host allocations
    float* h_data = (float*)malloc(data_bytes);

    srand(42);
    for (size_t i = 0; i < (size_t)M * N; ++i)
        h_data[i] = (float)(rand() % 1000) / 100.0f;

    // Device allocations
    float *d_data, *d_mean, *d_stddev, *d_corr;
    CUDA_CHECK(cudaMalloc(&d_data,   data_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean,   vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_stddev, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_corr,   corr_bytes));

    int threads1d  = block_size;
    int blocks_n   = (N + threads1d - 1) / threads1d;
    int blocks_mn  = (M * N + threads1d - 1) / threads1d;

    dim3 corr_block(TILE_SIZE, TILE_SIZE);
    dim3 corr_grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    char params[192];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"verification\":\"full_cpu_v1\"",
             N, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_mean, acc_stddev, acc_norm, acc_corr;

    // One full correlation pass (run exactly once). normalize_kernel mutates
    // d_data in place, so the pristine input is re-uploaded at the start of
    // the pass (untimed), keeping every execution identical.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_mean);
        bench_lap_reset(&acc_stddev);
        bench_lap_reset(&acc_norm);
        bench_lap_reset(&acc_corr);

        CUDA_CHECK(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_mean, bench_time_op(&timer, [&]() {
            mean_kernel<<<dim3(blocks_n), dim3(threads1d)>>>(d_data, d_mean, M, N);
        }));
        bench_lap_add(&acc_stddev, bench_time_op(&timer, [&]() {
            stddev_kernel<<<dim3(blocks_n), dim3(threads1d)>>>(d_data, d_mean, d_stddev, M, N);
        }));
        bench_lap_add(&acc_norm, bench_time_op(&timer, [&]() {
            normalize_kernel<<<dim3(blocks_mn), dim3(threads1d)>>>(d_data, d_mean, d_stddev, M, N);
        }));
        bench_lap_add(&acc_corr, bench_time_op(&timer, [&]() {
            correlation_kernel<<<corr_grid, corr_block>>>(d_data, d_corr, M, N);
        }));
    };

    run_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("mean_kernel",        0, &acc_mean,   params);
    bench_emit_iter("stddev_kernel",      0, &acc_stddev, params);
    bench_emit_iter("normalize_kernel",   0, &acc_norm,   params);
    bench_emit_iter("correlation_kernel", 0, &acc_corr,   params);
    total_gpu_ms += acc_mean.gpu_ms + acc_stddev.gpu_ms +
                    acc_norm.gpu_ms + acc_corr.gpu_ms;
    fprintf(stderr, "mean=%.4f  stddev=%.4f  normalize=%.4f  "
            "correlation=%.4f (gpu ms)\n",
            acc_mean.gpu_ms, acc_stddev.gpu_ms,
            acc_norm.gpu_ms, acc_corr.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify all four measured outputs after timing, with no extra GPU pass.
    const auto reference = cpu_correlation(h_data, M, N);
    const std::vector<double>* expected[] = {
        &reference.mean, &reference.stddev, &reference.normalized, &reference.correlation
    };
    float* outputs[] = {d_mean, d_stddev, d_data, d_corr};
    const char* names[] = {"mean", "stddev", "normalized", "correlation"};
    size_t mismatches = 0, checked = 0;
    for (int stage = 0; stage < 4; ++stage) {
        // M=N in this driver: each output fits the original host data buffer,
        // which the complete independent reference has already consumed.
        CUDA_CHECK(cudaMemcpy(h_data, outputs[stage], expected[stage]->size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < expected[stage]->size(); ++i) {
            const double value = (*expected[stage])[i];
            const bool diagonal = stage == 3 && i % ((size_t)N + 1) == 0;
            if (!correlation_value_matches(h_data[i], value) || (diagonal && h_data[i] != 1.0f)) {
                if (mismatches < 10)
                    fprintf(stderr, "MISMATCH %s[%zu]: GPU=%.9g CPU=%.17g\n",
                            names[stage], i, h_data[i], value);
                ++mismatches;
            }
            ++checked;
        }
    }
    fprintf(stderr, "Correlation verification: %s (%zu values, %zu mismatches)\n",
            mismatches ? "FAIL" : "PASS", checked, mismatches);

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_stddev));
    CUDA_CHECK(cudaFree(d_corr));
    free(h_data);

    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
