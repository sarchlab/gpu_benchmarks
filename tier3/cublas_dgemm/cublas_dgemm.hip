/**
 * cublas_dgemm.hip — FP64 GEMM benchmark using cuBLAS (NVIDIA) or rocBLAS (AMD).
 *
 * Computes: C = A * B
 *   where A is M×K, B is K×N, C is M×N (all row-major, FP64)
 *
 * Row-major GEMM via col-major BLAS trick:
 *   Row-major C = A*B  ↔  Col-major C^T = B^T * A^T
 *   cublasDgemm / rocblas_dgemm with op_none, op_none, dims (N, M, K)
 *   and leading dimensions: ldB=N, ldA=K, ldC=N
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   CUDA:   nvcc -O3 -std=c++14 -x cu -DUSE_CUBLAS  cublas_dgemm.hip -lcublas -o cublas_dgemm
 *   ROCm:   hipcc -O3 -std=c++14     -DUSE_ROCBLAS cublas_dgemm.hip -lrocblas -o cublas_dgemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef USE_CUBLAS
#  include <cublas_v2.h>
#  include <cuda_runtime.h>

#  define GPU_CHECK(x) do {                                                  \
       cudaError_t _e = (x);                                                 \
       if (_e != cudaSuccess) {                                               \
           fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#  define BLAS_CHECK(x) do {                                                 \
       cublasStatus_t _e = (x);                                              \
       if (_e != CUBLAS_STATUS_SUCCESS) {                                     \
           fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

typedef cudaEvent_t gpu_event_t;
#  define GPU_EVENT_CREATE(e)       GPU_CHECK(cudaEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)       GPU_CHECK(cudaEventRecord(e))
#  define GPU_EVENT_SYNC(e)         GPU_CHECK(cudaEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(cudaEventElapsedTime(&(ms),(a),(b)))
#  define GPU_DEVICE_SYNC()         GPU_CHECK(cudaDeviceSynchronize())

#elif defined(USE_ROCBLAS)
#  include <rocblas/rocblas.h>
#  include <hip/hip_runtime.h>

#  define GPU_CHECK(x) do {                                                  \
       hipError_t _e = (x);                                                  \
       if (_e != hipSuccess) {                                                \
           fprintf(stderr, "HIP error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, hipGetErrorString(_e));                \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#  define BLAS_CHECK(x) do {                                                 \
       rocblas_status _e = (x);                                              \
       if (_e != rocblas_status_success) {                                    \
           fprintf(stderr, "rocBLAS error %s:%d: %d\n",                      \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

typedef hipEvent_t gpu_event_t;
#  define GPU_EVENT_CREATE(e)       GPU_CHECK(hipEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)       GPU_CHECK(hipEventRecord((e), 0))
#  define GPU_EVENT_SYNC(e)         GPU_CHECK(hipEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(hipEventElapsedTime(&(ms),(a),(b)))
#  define GPU_DEVICE_SYNC()         GPU_CHECK(hipDeviceSynchronize())

#else
#  error "Define USE_CUBLAS or USE_ROCBLAS"
#endif

/* -------------------------------------------------------------------------
 * Benchmark parameters (defaults, overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int M      = 4096;
static int N      = 4096;
static int K      = 4096;
static int WARMUP = 0;
static int ITERS  = 5;

/* -------------------------------------------------------------------------
 * CPU reference DGEMM (only for a small corner — O(verify_m * verify_n * K))
 * ---------------------------------------------------------------------- */
static void cpu_dgemm(int vm, int vn, int vk,
                      const double * __restrict__ A,
                      const double * __restrict__ B,
                      double       * __restrict__ C)
{
    /* A is vm×vk (row-major, stride K)
     * B is vk×vn (row-major, stride N)
     * C is vm×vn (row-major, stride vn) */
    for (int i = 0; i < vm; i++) {
        for (int j = 0; j < vn; j++) {
            double s = 0.0;
            for (int p = 0; p < vk; p++)
                s += A[(size_t)i * K + p] * B[(size_t)p * N + j];
            C[i * vn + j] = s;
        }
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Read parameters from environment variables                          */
    /* ------------------------------------------------------------------ */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; K = M; }
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) ITERS = atoi(env_val);

    /* ------------------------------------------------------------------ */
    /* Allocate and initialize host matrices                               */
    /* ------------------------------------------------------------------ */
    double *hA = (double *)malloc((size_t)M * K * sizeof(double));
    double *hB = (double *)malloc((size_t)K * N * sizeof(double));
    double *hC = (double *)malloc((size_t)M * N * sizeof(double));

    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (double)rand() / (double)RAND_MAX;
    for (int i = 0; i < K * N; i++) hB[i] = (double)rand() / (double)RAND_MAX;
    memset(hC, 0, (size_t)M * N * sizeof(double));

    /* ------------------------------------------------------------------ */
    /* Device memory                                                        */
    /* ------------------------------------------------------------------ */
    double *dA, *dB, *dC;
    double alpha = 1.0, beta = 0.0;

    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha = atof(env_val);

#ifdef USE_CUBLAS
    GPU_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(double)));
    GPU_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(double)));
    GPU_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(double)));
    GPU_CHECK(cudaMemcpy(dA, hA, (size_t)M * K * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, hB, (size_t)K * N * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(double)));

    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));

#   define RUN_DGEMM() \
        BLAS_CHECK(cublasDgemm(handle,                              \
            CUBLAS_OP_N, CUBLAS_OP_N,                               \
            N, M, K,                                                \
            &alpha, dB, N, dA, K, &beta, dC, N))

#elif defined(USE_ROCBLAS)
    GPU_CHECK(hipMalloc(&dA, (size_t)M * K * sizeof(double)));
    GPU_CHECK(hipMalloc(&dB, (size_t)K * N * sizeof(double)));
    GPU_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(double)));
    GPU_CHECK(hipMemcpy(dA, hA, (size_t)M * K * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dB, hB, (size_t)K * N * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(double)));

    rocblas_handle handle;
    BLAS_CHECK(rocblas_create_handle(&handle));

#   define RUN_DGEMM() \
        BLAS_CHECK(rocblas_dgemm(handle,                                   \
            rocblas_operation_none, rocblas_operation_none,                 \
            N, M, K,                                                        \
            &alpha, dB, N, dA, K, &beta, dC, N))
#endif

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) {
        RUN_DGEMM();
    }
    GPU_DEVICE_SYNC();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    gpu_event_t ev_start, ev_stop;
    GPU_EVENT_CREATE(ev_start);
    GPU_EVENT_CREATE(ev_stop);

    GPU_EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) {
        RUN_DGEMM();
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)ITERS;

    /* ------------------------------------------------------------------ */
    /* Copy result back                                                    */
    /* ------------------------------------------------------------------ */
#ifdef USE_CUBLAS
    GPU_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(double), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
#elif defined(USE_ROCBLAS)
    GPU_CHECK(hipMemcpy(hC, dC, (size_t)M * N * sizeof(double), hipMemcpyDeviceToHost));
    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
#endif

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU DGEMM                               */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    double *cpuC = (double *)calloc((size_t)VN * VN, sizeof(double));
    cpu_dgemm(VN, VN, K, hA, hB, cpuC);

    /* cpuC is packed VN×VN (row stride = VN).
     * hC is full M×N (row stride = N).
     * Compare element (row i, col j) from each. */
    int pass = 1;
    for (int i = 0; i < VN && pass; i++) {
        for (int j = 0; j < VN && pass; j++) {
            double ref = cpuC[i * VN + j];
            double got = hC[(size_t)i * N + j];
            double err = fabs(ref - got) / (fabs(ref) + 1e-10);
            if (err > 1e-9) {
                fprintf(stderr, "MISMATCH at [%d,%d]: ref=%.15f got=%.15f rel=%.4e\n",
                        i, j, ref, got, err);
                pass = 0;
            }
        }
    }
    free(cpuC);
    fprintf(stderr, "Verification (first %d×%d corner): %s\n", VN, VN,
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Report — JSON-lines to stdout                                       */
    /* ------------------------------------------------------------------ */
    double gflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

#ifdef USE_CUBLAS
    printf("{\"type\":\"kernel\",\"name\":\"cublasDgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           (double)avg_ms, M, WARMUP, ITERS, alpha);
#elif defined(USE_ROCBLAS)
    printf("{\"type\":\"kernel\",\"name\":\"rocblas_dgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           (double)avg_ms, M, WARMUP, ITERS, alpha);
#endif

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, gflops);

    /* Human-readable to stderr */
    fprintf(stderr, "cublas_dgemm  M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            M, N, K, (double)avg_ms, gflops);

    free(hA); free(hB); free(hC);
    return pass ? 0 : 1;
}
