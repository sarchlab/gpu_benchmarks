/**
 * tensor_core_throughput.hip
 *
 * Measures Tensor Core throughput (TOPS — Tera Operations Per Second) using
 * the WMMA (Warp Matrix Multiply Accumulate) API.
 *
 * Compiles for both AMD (ROCm/HIP via rocWMMA) and NVIDIA (CUDA via
 * nvcuda::wmma) targets using ifdefs.
 *
 * Strategy:
 *   Perform a large tiled GEMM (D = A * B + C) using WMMA intrinsics with
 *   16×16×16 tiles.  The entire computation lives in registers / shared
 *   memory — we measure raw Tensor Core arithmetic throughput.
 *
 *   TOPS = 2 * M * N * K / time_s / 1e12
 *
 * Usage:
 *   ./tensor_core_throughput [matrix_dim]
 *
 *   matrix_dim  — M=N=K dimension (default 4096)
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_matrix_dim
 *   BENCH_PARAM_precision          ("fp16", "tf32", or "all")
 *   BENCH_PARAM_warmup
 *   BENCH_PARAM_timed_iterations
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Platform abstraction
// ---------------------------------------------------------------------------

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define gpuMalloc              cudaMalloc
#define gpuFree                cudaFree
#define gpuMemcpy              cudaMemcpy
#define gpuMemset              cudaMemset
#define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuEventCreate         cudaEventCreate
#define gpuEventRecord         cudaEventRecord
#define gpuEventSynchronize    cudaEventSynchronize
#define gpuEventElapsedTime    cudaEventElapsedTime
#define gpuEventDestroy        cudaEventDestroy
#define gpuGetDevice           cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties

typedef cudaEvent_t      gpuEvent_t;
typedef cudaDeviceProp   gpuDeviceProp_t;

#else  /* USE_ROCM (default) */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _err = (call);                                              \
        if (_err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define gpuMalloc              hipMalloc
#define gpuFree                hipFree
#define gpuMemcpy              hipMemcpy
#define gpuMemset              hipMemset
#define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define gpuDeviceSynchronize   hipDeviceSynchronize
#define gpuEventCreate         hipEventCreate
#define gpuEventRecord         hipEventRecord
#define gpuEventSynchronize    hipEventSynchronize
#define gpuEventElapsedTime    hipEventElapsedTime
#define gpuEventDestroy        hipEventDestroy
#define gpuGetDevice           hipGetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties

typedef hipEvent_t       gpuEvent_t;
typedef hipDeviceProp_t  gpuDeviceProp_t;

#endif

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_DIM               = 4096;
static const int WMMA_M                    = 16;
static const int WMMA_N                    = 16;
static const int WMMA_K                    = 16;
static const int WARP_SIZE                 = 32;
static const int DEFAULT_WARMUP_ITERATIONS = 0;
static const int DEFAULT_TIMED_ITERATIONS  = 5;

// ---------------------------------------------------------------------------
// FP16 WMMA GEMM kernel
// ---------------------------------------------------------------------------

__global__ void wmma_fp16_gemm(const half *A, const half *B, half *C,
                               int M, int N, int K)
{
    int warpM = blockIdx.x * WMMA_M;
    int warpN = blockIdx.y * WMMA_N;

#ifdef USE_CUDA
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N,
                            wmma::mem_row_major);
#else  /* USE_ROCM */
    rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                      rocwmma::half_t, rocwmma::row_major> a_frag;
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                      rocwmma::half_t, rocwmma::row_major> b_frag;
    rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                      rocwmma::half_t> c_frag;

    rocwmma::fill_fragment(c_frag, static_cast<rocwmma::half_t>(0));

    for (int k = 0; k < K; k += WMMA_K) {
        rocwmma::load_matrix_sync(a_frag,
            reinterpret_cast<const rocwmma::half_t*>(A + warpM * K + k), K);
        rocwmma::load_matrix_sync(b_frag,
            reinterpret_cast<const rocwmma::half_t*>(B + k * N + warpN), N);
        rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    rocwmma::store_matrix_sync(
        reinterpret_cast<rocwmma::half_t*>(C + warpM * N + warpN),
        c_frag, N, rocwmma::mem_row_major);
#endif
}

// ---------------------------------------------------------------------------
// TF32 WMMA GEMM kernel (CUDA sm_80+ only)
// ---------------------------------------------------------------------------

#ifdef USE_CUDA
__global__ void wmma_tf32_gemm(const float *A, const float *B, float *C,
                               int M, int N, int K)
{
    int warpM = blockIdx.x * WMMA_M;
    int warpN = blockIdx.y * WMMA_N;

    const int TF32_K = 8;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, TF32_K, wmma::precision::tf32,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, TF32_K, wmma::precision::tf32,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, TF32_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += TF32_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N,
                            wmma::mem_row_major);
}
#endif

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

/**
 * Run a kernel multiple times and measure average time.
 * Returns TOPS via out parameter.
 */
template <typename LaunchFn>
static void benchmark_kernel(const char *kernel_name, const char *precision_str,
                             int M, int N, int K,
                             int warmup_iters, int timed_iters,
                             int matrix_dim,
                             LaunchFn launch_fn,
                             double *out_avg_ms, double *out_tops)
{
    gpuEvent_t ev_start, ev_stop;
    GPU_CHECK(gpuEventCreate(&ev_start));
    GPU_CHECK(gpuEventCreate(&ev_stop));

    /* Warmup iterations. */
    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn();
    }
    GPU_CHECK(gpuDeviceSynchronize());

    /* Timed iterations. */
    GPU_CHECK(gpuEventRecord(ev_start, 0));
    for (int i = 0; i < timed_iters; ++i) {
        launch_fn();
    }
    GPU_CHECK(gpuEventRecord(ev_stop, 0));
    GPU_CHECK(gpuEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    GPU_CHECK(gpuEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    double avg_ms = (double)elapsed_ms / timed_iters;
    double avg_s  = avg_ms * 1e-3;

    /* TOPS = 2 * M * N * K / time_s / 1e12 */
    double ops  = 2.0 * (double)M * (double)N * (double)K;
    double tops = ops / avg_s / 1e12;

    fprintf(stderr, "%-8s  %.3f ms  %.3f TOPS\n", precision_str, avg_ms, tops);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"precision\":\"%s\","
           "\"warmup_iterations\":%d,\"timed_iterations\":%d}}\n",
           kernel_name, avg_ms, matrix_dim, precision_str,
           warmup_iters, timed_iters);

    *out_avg_ms = avg_ms;
    *out_tops   = tops;

    GPU_CHECK(gpuEventDestroy(ev_start));
    GPU_CHECK(gpuEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters: env vars take priority, then argv, then defaults. */
    int dim = DEFAULT_DIM;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) {
        dim = atoi(env_val);
    } else if (argc >= 2) {
        dim = atoi(argv[1]);
    }
    if (dim <= 0 || (dim % WMMA_M) != 0) {
        fprintf(stderr, "Matrix dim must be a positive multiple of %d\n",
                WMMA_M);
        return EXIT_FAILURE;
    }

    const char* precision = "fp16";
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) precision = env_val;

    int warmup_iterations = DEFAULT_WARMUP_ITERATIONS;
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) warmup_iterations = atoi(env_val);

    int timed_iterations = DEFAULT_TIMED_ITERATIONS;
    env_val = getenv("BENCH_PARAM_timed_iterations");
    if (env_val) timed_iterations = atoi(env_val);

    int M = dim, N = dim, K = dim;

    /* Print device info to stderr. */
    int device_id = 0;
    GPU_CHECK(gpuGetDevice(&device_id));
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device           : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix (M×N×K)   : %d × %d × %d\n", M, N, K);
    fprintf(stderr, "WMMA tile        : %d × %d × %d\n", WMMA_M, WMMA_N, WMMA_K);
    fprintf(stderr, "Precision        : %s\n", precision);
    fprintf(stderr, "Warmup iters     : %d\n", warmup_iterations);
    fprintf(stderr, "Timed iters      : %d\n\n", timed_iterations);

    double total_time_ms = 0.0;
    double max_tops = 0.0;

    bool run_fp16 = (strcmp(precision, "fp16") == 0 || strcmp(precision, "all") == 0);
    bool run_tf32 = (strcmp(precision, "tf32") == 0 || strcmp(precision, "all") == 0);

    /* --------------------------------------------------------------------- */
    /* FP16 benchmark                                                        */
    /* --------------------------------------------------------------------- */
    if (run_fp16) {
        size_t mat_size = (size_t)M * K * sizeof(half);
        size_t out_size = (size_t)M * N * sizeof(half);

        half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        GPU_CHECK(gpuMalloc((void **)&d_A, mat_size));
        GPU_CHECK(gpuMalloc((void **)&d_B, (size_t)K * N * sizeof(half)));
        GPU_CHECK(gpuMalloc((void **)&d_C, out_size));

        GPU_CHECK(gpuMemset(d_A, 0, mat_size));
        GPU_CHECK(gpuMemset(d_B, 0, (size_t)K * N * sizeof(half)));
        GPU_CHECK(gpuMemset(d_C, 0, out_size));

        dim3 grid(M / WMMA_M, N / WMMA_N);
        dim3 block(WARP_SIZE);

        double avg_ms, tops;
        benchmark_kernel("wmma_fp16_gemm", "fp16", M, N, K,
                         warmup_iterations, timed_iterations, dim,
                         [&]() {
            wmma_fp16_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }, &avg_ms, &tops);

        total_time_ms += avg_ms;
        if (tops > max_tops) max_tops = tops;

        GPU_CHECK(gpuFree(d_A));
        GPU_CHECK(gpuFree(d_B));
        GPU_CHECK(gpuFree(d_C));
    }

    /* --------------------------------------------------------------------- */
    /* TF32 benchmark (CUDA Ampere+ only)                                    */
    /* --------------------------------------------------------------------- */
#ifdef USE_CUDA
    if (run_tf32) {
        int major = prop.major;
        if (major >= 8) {
            size_t mat_size = (size_t)M * K * sizeof(float);
            size_t out_size = (size_t)M * N * sizeof(float);

            float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
            GPU_CHECK(gpuMalloc((void **)&d_A, mat_size));
            GPU_CHECK(gpuMalloc((void **)&d_B, (size_t)K * N * sizeof(float)));
            GPU_CHECK(gpuMalloc((void **)&d_C, out_size));
            GPU_CHECK(gpuMemset(d_A, 0, mat_size));
            GPU_CHECK(gpuMemset(d_B, 0, (size_t)K * N * sizeof(float)));
            GPU_CHECK(gpuMemset(d_C, 0, out_size));

            dim3 grid(M / WMMA_M, N / WMMA_N);
            dim3 block(WARP_SIZE);

            double avg_ms, tops;
            benchmark_kernel("wmma_tf32_gemm", "tf32", M, N, K,
                             warmup_iterations, timed_iterations, dim,
                             [&]() {
                wmma_tf32_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            }, &avg_ms, &tops);

            total_time_ms += avg_ms;
            if (tops > max_tops) max_tops = tops;

            GPU_CHECK(gpuFree(d_A));
            GPU_CHECK(gpuFree(d_B));
            GPU_CHECK(gpuFree(d_C));
        } else {
            fprintf(stderr, "TF32 not supported (requires sm_80+, have sm_%d%d)\n",
                   prop.major, prop.minor);
        }
    }
#else
    if (run_tf32) {
        fprintf(stderr, "TF32 not available on ROCm (FP16 only via rocWMMA)\n");
    }
#endif

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"TFLOPS\",\"value\":%.3f}]}\n",
           total_time_ms, max_tops);

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
