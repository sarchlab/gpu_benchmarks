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
 *   Tests both FP16 (half) and, on NVIDIA Ampere+, TF32 precision.
 *
 * Matrix sizes:
 *   M = N = K = 4096 (default), yielding ~137.4 GFLOP per GEMM.
 *
 * Timing:
 *   5 warmup + 5 timed iterations; report average.
 *
 * CSV output (to stdout):
 *   tensor_core_throughput,fp16,<time_ms>,<TOPS>
 *   tensor_core_throughput,tf32,<time_ms>,<TOPS>   (CUDA sm_80+ only)
 *
 * Usage:
 *   ./tensor_core_throughput [matrix_dim]
 *
 *   matrix_dim  — M=N=K dimension (default 4096)
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
// Constants
// ---------------------------------------------------------------------------

static const int DEFAULT_DIM         = 4096;
static const int WMMA_M              = 16;
static const int WMMA_N              = 16;
static const int WMMA_K              = 16;
static const int WARP_SIZE           = 32;
static const int WARMUP_ITERATIONS   = 5;
static const int TIMED_ITERATIONS    = 5;

// ---------------------------------------------------------------------------
// FP16 WMMA GEMM kernel
// ---------------------------------------------------------------------------

/**
 * Tiled GEMM using WMMA 16×16×16 FP16 fragments.
 *
 * Grid:  (M / WMMA_M, N / WMMA_N) blocks, each block = 1 warp (32 threads).
 * Each warp computes one 16×16 output tile by iterating over K in steps of
 * WMMA_K.
 *
 * @param A   Row-major FP16 matrix [M × K]
 * @param B   Row-major FP16 matrix [K × N]
 * @param C   Row-major FP16 matrix [M × N] (output)
 * @param M   Matrix rows
 * @param N   Matrix columns
 * @param K   Inner dimension
 */
__global__ void wmma_fp16_gemm(const half *A, const half *B, half *C,
                               int M, int N, int K)
{
    /* Each block is one warp responsible for one 16×16 output tile. */
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

    /* TF32: uses wmma::precision::tf32 on Ampere+.
       Fragment shapes: 16×16×8 for TF32 on NVIDIA. */
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
 *
 * @param label        Precision label for CSV output ("fp16" or "tf32")
 * @param M            Matrix dimension M
 * @param N            Matrix dimension N
 * @param K            Matrix dimension K
 * @param launch_fn    Lambda that launches the kernel
 */
template <typename LaunchFn>
static void benchmark_kernel(const char *label, int M, int N, int K,
                             LaunchFn launch_fn)
{
    gpuEvent_t ev_start, ev_stop;
    GPU_CHECK(gpuEventCreate(&ev_start));
    GPU_CHECK(gpuEventCreate(&ev_stop));

    /* Warmup iterations. */
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        launch_fn();
    }
    GPU_CHECK(gpuDeviceSynchronize());

    /* Timed iterations. */
    GPU_CHECK(gpuEventRecord(ev_start, 0));
    for (int i = 0; i < TIMED_ITERATIONS; ++i) {
        launch_fn();
    }
    GPU_CHECK(gpuEventRecord(ev_stop, 0));
    GPU_CHECK(gpuEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    GPU_CHECK(gpuEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    double avg_ms = (double)elapsed_ms / TIMED_ITERATIONS;
    double avg_s  = avg_ms * 1e-3;

    /* TOPS = 2 * M * N * K / time_s / 1e12 */
    double ops  = 2.0 * (double)M * (double)N * (double)K;
    double tops = ops / avg_s / 1e12;

    printf("%-8s  %10.3f ms  %8.3f TOPS\n", label, avg_ms, tops);

    /* CSV: tensor_core_throughput,<dtype>,<time_ms>,<TOPS> */
    printf("tensor_core_throughput,%s,%.3f,%.3f\n", label, avg_ms, tops);

    GPU_CHECK(gpuEventDestroy(ev_start));
    GPU_CHECK(gpuEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int dim = DEFAULT_DIM;
    if (argc >= 2) {
        dim = atoi(argv[1]);
        if (dim <= 0 || (dim % WMMA_M) != 0) {
            fprintf(stderr, "Matrix dim must be a positive multiple of %d\n",
                    WMMA_M);
            return EXIT_FAILURE;
        }
    }

    int M = dim, N = dim, K = dim;

    /* Print device info. */
    int device_id = 0;
    GPU_CHECK(gpuGetDevice(&device_id));
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, device_id));

    printf("Device           : %s (id=%d)\n", prop.name, device_id);
    printf("Matrix (M×N×K)   : %d × %d × %d\n", M, N, K);
    printf("WMMA tile        : %d × %d × %d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Warmup iters     : %d\n", WARMUP_ITERATIONS);
    printf("Timed iters      : %d\n", TIMED_ITERATIONS);
    printf("\n");

    /* --------------------------------------------------------------------- */
    /* FP16 benchmark                                                        */
    /* --------------------------------------------------------------------- */
    {
        size_t mat_size = (size_t)M * K * sizeof(half);
        size_t out_size = (size_t)M * N * sizeof(half);

        half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        GPU_CHECK(gpuMalloc((void **)&d_A, mat_size));
        GPU_CHECK(gpuMalloc((void **)&d_B, (size_t)K * N * sizeof(half)));
        GPU_CHECK(gpuMalloc((void **)&d_C, out_size));

        /* Initialize to small random-ish values (fill with 1.0h for simplicity). */
        GPU_CHECK(gpuMemset(d_A, 0, mat_size));
        GPU_CHECK(gpuMemset(d_B, 0, (size_t)K * N * sizeof(half)));
        GPU_CHECK(gpuMemset(d_C, 0, out_size));

        /* Grid: one warp per 16×16 output tile. */
        dim3 grid(M / WMMA_M, N / WMMA_N);
        dim3 block(WARP_SIZE);

        benchmark_kernel("fp16", M, N, K, [&]() {
            wmma_fp16_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        });

        GPU_CHECK(gpuFree(d_A));
        GPU_CHECK(gpuFree(d_B));
        GPU_CHECK(gpuFree(d_C));
    }

    /* --------------------------------------------------------------------- */
    /* TF32 benchmark (CUDA Ampere+ only)                                    */
    /* --------------------------------------------------------------------- */
#ifdef USE_CUDA
    {
        /* Check for sm_80+ (Ampere or later) for TF32 support. */
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

            benchmark_kernel("tf32", M, N, K, [&]() {
                wmma_tf32_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            });

            GPU_CHECK(gpuFree(d_A));
            GPU_CHECK(gpuFree(d_B));
            GPU_CHECK(gpuFree(d_C));
        } else {
            printf("TF32 not supported (requires sm_80+, have sm_%d%d)\n",
                   prop.major, prop.minor);
        }
    }
#else
    printf("TF32 not available on ROCm (FP16 only via rocWMMA)\n");
#endif

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
