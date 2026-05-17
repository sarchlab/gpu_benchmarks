/**
 * cublas_batched_gemm.hip — Batched FP32 GEMM using cuBLAS (NVIDIA) or rocBLAS (AMD).
 *
 * Computes: C[i] = alpha * A[i] * B[i] + beta * C[i]  for i in [0, batch_count)
 *   where each A[i] is M×K, B[i] is K×N, C[i] is M×N (all row-major, FP32)
 *   batch_count = 64, M = N = K = 512, alpha = 1.0, beta = 0.0
 *
 * Row-major GEMM via col-major BLAS trick:
 *   Row-major C = A*B  ↔  Col-major C^T = B^T * A^T
 *   cublasSgemmBatched / rocblas_sgemm_batched with op_none, op_none,
 *   dims (N, M, K) and leading dimensions: ldB=N, ldA=K, ldC=N
 *
 * Warmup: configurable iterations (not timed)
 * Timed:  single iteration (repetition handled by CLI)
 *
 * Verification: compare first 8×8 corner of C[0] against CPU GEMM
 * GFLOPS = batch_count * 2 * M * N * K / time_s / 1e9
 *
 * Output:
 *   cublas_batched_gemm,64x512,<time_ms>,<GFLOPS>
 *
 * Build:
 *   CUDA:   nvcc -O3 -std=c++14 -x cu -DUSE_CUBLAS  cublas_batched_gemm.hip -lcublas -o cublas_batched_gemm
 *   ROCm:   hipcc -O3 -std=c++14     -DUSE_ROCBLAS cublas_batched_gemm.hip -lrocblas -o cublas_batched_gemm
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
#  define GPU_EVENT_CREATE(e)     GPU_CHECK(cudaEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)     GPU_CHECK(cudaEventRecord(e))
#  define GPU_EVENT_SYNC(e)       GPU_CHECK(cudaEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(cudaEventElapsedTime(&(ms),(a),(b)))
#  define GPU_DEVICE_SYNC()       GPU_CHECK(cudaDeviceSynchronize())

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
#  define GPU_EVENT_CREATE(e)     GPU_CHECK(hipEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)     GPU_CHECK(hipEventRecord((e), 0))
#  define GPU_EVENT_SYNC(e)       GPU_CHECK(hipEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(hipEventElapsedTime(&(ms),(a),(b)))
#  define GPU_DEVICE_SYNC()       GPU_CHECK(hipDeviceSynchronize())

#else
#  error "Define USE_CUBLAS or USE_ROCBLAS"
#endif

/* -------------------------------------------------------------------------
 * Benchmark parameters (runtime-configurable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int BATCH_COUNT = 64;
static int M           = 512;
static int N           = 512;
static int K           = 512;
static int WARMUP      = 0;
/* -------------------------------------------------------------------------
 * CPU reference GEMM (only for a small corner — O(verify_m * verify_n * K))
 * A is vm×vk (row-major, stride K)
 * B is vk×vn (row-major, stride N)
 * C is vm×vn (row-major, stride vn)
 * ---------------------------------------------------------------------- */
static void cpu_sgemm(int vm, int vn, int vk,
                      const float * __restrict__ A,
                      const float * __restrict__ B,
                      float       * __restrict__ C)
{
    for (int i = 0; i < vm; i++) {
        for (int j = 0; j < vn; j++) {
            float s = 0.0f;
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

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; K = M; }
    env_val = getenv("BENCH_PARAM_batch_count");
    if (env_val) BATCH_COUNT = atoi(env_val);
    const size_t sizeA = (size_t)M * K;
    const size_t sizeB = (size_t)K * N;
    const size_t sizeC = (size_t)M * N;

    /* ------------------------------------------------------------------ */
    /* Allocate and initialize host matrices for the entire batch          */
    /* ------------------------------------------------------------------ */
    float *hA = (float *)malloc(BATCH_COUNT * sizeA * sizeof(float));
    float *hB = (float *)malloc(BATCH_COUNT * sizeB * sizeof(float));
    float *hC = (float *)malloc(BATCH_COUNT * sizeC * sizeof(float));

    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < BATCH_COUNT * sizeA; i++)
        hA[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < BATCH_COUNT * sizeB; i++)
        hB[i] = (float)rand() / (float)RAND_MAX;
    memset(hC, 0, BATCH_COUNT * sizeC * sizeof(float));

    /* ------------------------------------------------------------------ */
    /* Device memory — one large allocation per matrix, sliced by batch   */
    /* ------------------------------------------------------------------ */
    float *dA, *dB, *dC;

    /* Host arrays of device pointers for batched BLAS calls */
    float **h_Aptrs = (float **)malloc(BATCH_COUNT * sizeof(float *));
    float **h_Bptrs = (float **)malloc(BATCH_COUNT * sizeof(float *));
    float **h_Cptrs = (float **)malloc(BATCH_COUNT * sizeof(float *));

    /* Device arrays of device pointers */
    float **d_Aptrs, **d_Bptrs, **d_Cptrs;

#ifdef USE_CUBLAS
    GPU_CHECK(cudaMalloc(&dA, BATCH_COUNT * sizeA * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dB, BATCH_COUNT * sizeB * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dC, BATCH_COUNT * sizeC * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dA, hA, BATCH_COUNT * sizeA * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, hB, BATCH_COUNT * sizeB * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(dC, 0,  BATCH_COUNT * sizeC * sizeof(float)));

    /* Set up pointer arrays */
    for (int b = 0; b < BATCH_COUNT; b++) {
        h_Aptrs[b] = dA + b * sizeA;
        h_Bptrs[b] = dB + b * sizeB;
        h_Cptrs[b] = dC + b * sizeC;
    }

    GPU_CHECK(cudaMalloc(&d_Aptrs, BATCH_COUNT * sizeof(float *)));
    GPU_CHECK(cudaMalloc(&d_Bptrs, BATCH_COUNT * sizeof(float *)));
    GPU_CHECK(cudaMalloc(&d_Cptrs, BATCH_COUNT * sizeof(float *)));
    GPU_CHECK(cudaMemcpy(d_Aptrs, h_Aptrs, BATCH_COUNT * sizeof(float *), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_Bptrs, h_Bptrs, BATCH_COUNT * sizeof(float *), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_Cptrs, h_Cptrs, BATCH_COUNT * sizeof(float *), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    /* Row-major trick: C = A*B in row-major ↔ C^T = B^T * A^T in col-major
     * So we swap A/B pointers and use dims (N, M, K). */
#   define RUN_BATCHED_GEMM() \
        BLAS_CHECK(cublasSgemmBatched(handle,                           \
            CUBLAS_OP_N, CUBLAS_OP_N,                                   \
            N, M, K,                                                    \
            &alpha,                                                     \
            (const float **)d_Bptrs, N,                                 \
            (const float **)d_Aptrs, K,                                 \
            &beta,                                                      \
            d_Cptrs, N,                                                 \
            BATCH_COUNT))

#elif defined(USE_ROCBLAS)
    GPU_CHECK(hipMalloc(&dA, BATCH_COUNT * sizeA * sizeof(float)));
    GPU_CHECK(hipMalloc(&dB, BATCH_COUNT * sizeB * sizeof(float)));
    GPU_CHECK(hipMalloc(&dC, BATCH_COUNT * sizeC * sizeof(float)));
    GPU_CHECK(hipMemcpy(dA, hA, BATCH_COUNT * sizeA * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dB, hB, BATCH_COUNT * sizeB * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(dC, 0,  BATCH_COUNT * sizeC * sizeof(float)));

    /* Set up pointer arrays */
    for (int b = 0; b < BATCH_COUNT; b++) {
        h_Aptrs[b] = dA + b * sizeA;
        h_Bptrs[b] = dB + b * sizeB;
        h_Cptrs[b] = dC + b * sizeC;
    }

    GPU_CHECK(hipMalloc(&d_Aptrs, BATCH_COUNT * sizeof(float *)));
    GPU_CHECK(hipMalloc(&d_Bptrs, BATCH_COUNT * sizeof(float *)));
    GPU_CHECK(hipMalloc(&d_Cptrs, BATCH_COUNT * sizeof(float *)));
    GPU_CHECK(hipMemcpy(d_Aptrs, h_Aptrs, BATCH_COUNT * sizeof(float *), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_Bptrs, h_Bptrs, BATCH_COUNT * sizeof(float *), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_Cptrs, h_Cptrs, BATCH_COUNT * sizeof(float *), hipMemcpyHostToDevice));

    rocblas_handle handle;
    BLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;

    /* Row-major trick: swap A/B pointer arrays, use dims (N, M, K). */
#   define RUN_BATCHED_GEMM() \
        BLAS_CHECK(rocblas_sgemm_batched(handle,                               \
            rocblas_operation_none, rocblas_operation_none,                     \
            N, M, K,                                                           \
            &alpha,                                                            \
            (const float *const *)d_Bptrs, N,                                  \
            (const float *const *)d_Aptrs, K,                                  \
            &beta,                                                             \
            d_Cptrs, N,                                                        \
            BATCH_COUNT))
#endif

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) {
        RUN_BATCHED_GEMM();
    }
    GPU_DEVICE_SYNC();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    gpu_event_t ev_start, ev_stop;
    GPU_EVENT_CREATE(ev_start);
    GPU_EVENT_CREATE(ev_stop);

    GPU_EVENT_RECORD(ev_start);
    RUN_BATCHED_GEMM();
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms;

    /* ------------------------------------------------------------------ */
    /* Copy result back                                                    */
    /* ------------------------------------------------------------------ */
#ifdef USE_CUBLAS
    GPU_CHECK(cudaMemcpy(hC, dC, BATCH_COUNT * sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(d_Aptrs); cudaFree(d_Bptrs); cudaFree(d_Cptrs);
#elif defined(USE_ROCBLAS)
    GPU_CHECK(hipMemcpy(hC, dC, BATCH_COUNT * sizeC * sizeof(float), hipMemcpyDeviceToHost));
    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    hipFree(d_Aptrs); hipFree(d_Bptrs); hipFree(d_Cptrs);
#endif

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner of C[0] vs CPU SGEMM                       */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    float *cpuC = (float *)calloc((size_t)VN * VN, sizeof(float));
    /* Use A[0] and B[0] for verification */
    cpu_sgemm(VN, VN, K, hA, hB, cpuC);

    /* cpuC is packed VN×VN (row stride = VN).
     * hC is first batch C[0], full M×N (row stride = N). */
    int pass = 1;
    for (int i = 0; i < VN && pass; i++) {
        for (int j = 0; j < VN && pass; j++) {
            float ref = cpuC[i * VN + j];
            float got = hC[(size_t)i * N + j];
            float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
            if (err > 1e-3f) {
                fprintf(stderr, "MISMATCH at [%d,%d]: ref=%.6f got=%.6f rel=%.4e\n",
                        i, j, ref, got, err);
                pass = 0;
            }
        }
    }
    free(cpuC);
    fprintf(stderr, "Verification (first %d×%d corner of batch 0): %s\n", VN, VN,
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    double gflops = (double)BATCH_COUNT * 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines output */
#ifdef USE_CUBLAS
    printf("{\"type\":\"kernel\",\"name\":\"cublasSgemmBatched\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"batch_count\":%d,\"warmup\":%d}}\n",
           (double)avg_ms, M, BATCH_COUNT, WARMUP);
#elif defined(USE_ROCBLAS)
    printf("{\"type\":\"kernel\",\"name\":\"rocblas_sgemm_batched\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"batch_count\":%d,\"warmup\":%d}}\n",
           (double)avg_ms, M, BATCH_COUNT, WARMUP);
#endif
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, gflops);

    /* Human-readable */
    fprintf(stderr, "cublas_batched_gemm  batch=%d M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            BATCH_COUNT, M, N, K, avg_ms, gflops);

    free(hA); free(hB); free(hC);
    free(h_Aptrs); free(h_Bptrs); free(h_Cptrs);
    return pass ? 0 : 1;
}
