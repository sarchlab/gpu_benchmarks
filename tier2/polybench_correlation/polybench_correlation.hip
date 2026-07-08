// polybench_correlation.hip — PolyBench Correlation benchmark (HIP, self-contained)
//
// Compute correlation matrix from a data matrix (M samples x N features).
// Steps: mean, stddev, normalize, correlation (matmul-like).
// Derived from the PolyBench/GPU benchmark suite.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_precision
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (mean_kernel, stddev_kernel, normalize_kernel, correlation_kernel)
// carrying that kernel's GPU event time and host time within the pass, then
// a summary whose total_time_ms is the SUM of all recorded GPU event times.
// Derived metrics are computed downstream from time_ms and params.
// Output (stderr): device info, timing details

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
    const char* precision = "float";

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) {
        // Validate against the supported set; params must carry canonical
        // literals, never raw env text.
        if (strcmp(env_val, "float") != 0) {
            fprintf(stderr, "Unsupported BENCH_PARAM_precision '%s' (supported: float)\n",
                    env_val);
            return EXIT_FAILURE;
        }
        precision = "float";
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int M = N;

    size_t data_bytes = (size_t)M * N * sizeof(float);
    size_t vec_bytes  = (size_t)N * sizeof(float);
    size_t corr_bytes = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Data matrix: %dx%d\n\n", M, N);

    // Host allocations
    float* h_data = (float*)malloc(data_bytes);

    srand(42);
    for (size_t i = 0; i < (size_t)M * N; ++i)
        h_data[i] = (float)(rand() % 1000) / 100.0f;

    // Device allocations
    float *d_data, *d_mean, *d_stddev, *d_corr;
    HIP_CHECK(hipMalloc(&d_data,   data_bytes));
    HIP_CHECK(hipMalloc(&d_mean,   vec_bytes));
    HIP_CHECK(hipMalloc(&d_stddev, vec_bytes));
    HIP_CHECK(hipMalloc(&d_corr,   corr_bytes));

    int threads1d  = block_size;
    int blocks_n   = (N + threads1d - 1) / threads1d;
    int blocks_mn  = (M * N + threads1d - 1) / threads1d;

    dim3 corr_block(TILE_SIZE, TILE_SIZE);
    dim3 corr_grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"precision\":\"%s\"",
             N, block_size, precision);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_mean, acc_stddev, acc_norm, acc_corr;

    // One full correlation pass. normalize_kernel mutates d_data in place, so
    // the pristine input is re-uploaded at the start of every pass (untimed),
    // exactly as the pre-conversion code did.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_mean);
        bench_lap_reset(&acc_stddev);
        bench_lap_reset(&acc_norm);
        bench_lap_reset(&acc_corr);

        HIP_CHECK(hipMemcpy(d_data, h_data, data_bytes, hipMemcpyHostToDevice));

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

    for (int w = 0; w < num_warmup; ++w) {
        run_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_pass();
        bench_emit_iter("mean_kernel",        i, &acc_mean,   params);
        bench_emit_iter("stddev_kernel",      i, &acc_stddev, params);
        bench_emit_iter("normalize_kernel",   i, &acc_norm,   params);
        bench_emit_iter("correlation_kernel", i, &acc_corr,   params);
        total_gpu_ms += acc_mean.gpu_ms + acc_stddev.gpu_ms +
                        acc_norm.gpu_ms + acc_corr.gpu_ms;
        fprintf(stderr, "pass %d: mean=%.4f  stddev=%.4f  normalize=%.4f  "
                "correlation=%.4f (gpu ms)\n",
                i, acc_mean.gpu_ms, acc_stddev.gpu_ms,
                acc_norm.gpu_ms, acc_corr.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_mean));
    HIP_CHECK(hipFree(d_stddev));
    HIP_CHECK(hipFree(d_corr));
    free(h_data);

    return 0;
}
