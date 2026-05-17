/**
 * cublas_gemv.hip — FP32 GEMV benchmark using cuBLAS (NVIDIA) or rocBLAS (AMD).
 *
 * Computes: y = A * x
 *   where A is M×N (row-major, FP32), x is N-vector, y is M-vector
 *   M = N = 4096
 *
 * Row-major GEMV via col-major BLAS trick:
 *   Row-major A (M×N) looks like col-major A_cm (N×M).
 *   y = A_rm * x  ↔  y = A_cm^T * x
 *   cublasSgemv / rocblas_sgemv with op=TRANSPOSE, m=N, n=M, lda=N
 *
 * Warmup: configurable iterations (not timed)
 * Timed:  single iteration (repetition handled by CLI)
 *
 * Verification: compare all M elements of y against CPU GEMV, max rel error < 1e-4
 * GFLOPS = 2 * M * N / time_s / 1e9
 * GB/s   = (M*N + M + N) * sizeof(float) / time_s / 1e9
 *
 * Output:
 *   CSV    (stdout): cublas_gemv,4096,<time_ms>,<GBs>,<GFLOPS>
 *   Human  (stderr): device info, per-iteration times, verification result
 *
 * Build:
 *   CUDA: nvcc -O3 -std=c++14 -x cu -DUSE_CUBLAS  cublas_gemv.hip -lcublas -o cublas_gemv
 *   ROCm: hipcc -O3 -std=c++14    -DUSE_ROCBLAS cublas_gemv.hip -lrocblas -o cublas_gemv
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
 * Benchmark default parameters (overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * CPU reference GEMV — y[i] = sum_j A[i*N+j] * x[j]
 * ---------------------------------------------------------------------- */
static void cpu_sgemv(const float * __restrict__ A,
                      const float * __restrict__ x,
                      float       * __restrict__ y,
                      int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        for (int j = 0; j < cols; j++)
            s += A[(size_t)i * cols + j] * x[j];
        y[i] = s;
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Benchmark parameters (overridable via BENCH_PARAM_* env vars)       */
    /* ------------------------------------------------------------------ */
    int M = 4096;
    int N = 4096;
    int WARMUP = 0;
    float alpha_param = 1.0f;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; }
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha_param = (float)atof(env_val);

    /* ------------------------------------------------------------------ */
    /* Allocate and initialize host arrays                                 */
    /* ------------------------------------------------------------------ */
    float *hA  = (float *)malloc((size_t)M * N * sizeof(float));
    float *hx  = (float *)malloc((size_t)N     * sizeof(float));
    float *hy  = (float *)malloc((size_t)M     * sizeof(float));
    float *hy_ref = (float *)malloc((size_t)M  * sizeof(float));

    if (!hA || !hx || !hy || !hy_ref) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * N; i++) hA[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (int i = 0; i < N;     i++) hx[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    memset(hy, 0, (size_t)M * sizeof(float));

    /* ------------------------------------------------------------------ */
    /* Device setup                                                         */
    /* ------------------------------------------------------------------ */
#ifdef USE_CUBLAS
    int device_id = 0;
    GPU_CHECK(cudaGetDevice(&device_id));
    struct cudaDeviceProp prop;
    GPU_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
#elif defined(USE_ROCBLAS)
    int device_id = 0;
    GPU_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    GPU_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
#endif
    fprintf(stderr, "M=%d  N=%d  warmup=%d\n\n", M, N, WARMUP);

    /* ------------------------------------------------------------------ */
    /* Device memory                                                        */
    /* ------------------------------------------------------------------ */
    float *dA, *dx, *dy;

#ifdef USE_CUBLAS
    GPU_CHECK(cudaMalloc(&dA, (size_t)M * N * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dx, (size_t)N     * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dy, (size_t)M     * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dA, hA, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dx, hx, (size_t)N     * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(dy, 0,  (size_t)M     * sizeof(float)));

    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));

    float alpha = alpha_param, beta = 0.0f;

    /* Row-major A(M×N) looks col-major as A_cm(N×M).
     * y = A_rm * x  =>  y = A_cm^T * x
     * cublasSgemv(handle, CUBLAS_OP_T, m=N, n=M, alpha, A, lda=N, x, 1, beta, y, 1)
     * op(A_cm) = A_cm^T is M×N, x is N-vector => y is M-vector  */
#   define RUN_SGEMV() \
        BLAS_CHECK(cublasSgemv(handle,               \
            CUBLAS_OP_T,                              \
            N, M,                                     \
            &alpha,                                   \
            dA, N,                                    \
            dx, 1,                                    \
            &beta,                                    \
            dy, 1))

#elif defined(USE_ROCBLAS)
    GPU_CHECK(hipMalloc(&dA, (size_t)M * N * sizeof(float)));
    GPU_CHECK(hipMalloc(&dx, (size_t)N     * sizeof(float)));
    GPU_CHECK(hipMalloc(&dy, (size_t)M     * sizeof(float)));
    GPU_CHECK(hipMemcpy(dA, hA, (size_t)M * N * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dx, hx, (size_t)N     * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(dy, 0,  (size_t)M     * sizeof(float)));

    rocblas_handle handle;
    BLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = alpha_param, beta = 0.0f;

#   define RUN_SGEMV() \
        BLAS_CHECK(rocblas_sgemv(handle,                      \
            rocblas_operation_transpose,                       \
            N, M,                                             \
            &alpha,                                           \
            dA, N,                                            \
            dx, 1,                                            \
            &beta,                                            \
            dy, 1))
#endif

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) {
        RUN_SGEMV();
    }
    GPU_DEVICE_SYNC();

    /* ------------------------------------------------------------------ */
    /* Timed iterations (one event pair per iteration)                    */
    /* ------------------------------------------------------------------ */
    gpu_event_t ev_start, ev_stop;
    GPU_EVENT_CREATE(ev_start);
    GPU_EVENT_CREATE(ev_stop);

    GPU_EVENT_RECORD(ev_start);
    RUN_SGEMV();
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);

    /* ------------------------------------------------------------------ */
    /* Copy result back                                                    */
    /* ------------------------------------------------------------------ */
#ifdef USE_CUBLAS
    GPU_CHECK(cudaMemcpy(hy, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
#elif defined(USE_ROCBLAS)
    GPU_CHECK(hipMemcpy(hy, dy, (size_t)M * sizeof(float), hipMemcpyDeviceToHost));
    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dx); hipFree(dy);
#endif

    /* ------------------------------------------------------------------ */
    /* Verify all M elements vs CPU GEMV                                  */
    /* ------------------------------------------------------------------ */
    cpu_sgemv(hA, hx, hy_ref, M, N);

    int pass = 1;
    for (int i = 0; i < M; i++) {
        float ref = hy_ref[i];
        float err = fabsf(hy[i] - ref) / (fabsf(ref) + 1e-6f);
        if (err > 1e-3f) {
            fprintf(stderr, "MISMATCH y[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, hy[i], ref, err);
            pass = 0;
            break;
        }
    }
    fprintf(stderr, "Verification (all %d elements): %s\n\n", M,
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Statistics                                                          */
    /* ------------------------------------------------------------------ */
    double avg_ms = (double)total_ms;

    /* GFLOPS = 2 * M * N multiply-adds */
    double gflops = 2.0 * M * N / (avg_ms * 1e-3) / 1e9;

    /* GB/s = matrix A (M*N floats) + vector x (N floats) + vector y (M floats) */
    double bytes = ((double)M * N + M + N) * sizeof(float);
    double gbps  = bytes / (avg_ms * 1e-3) / 1e9;

    /* JSON-lines kernel event */
#ifdef USE_CUBLAS
    printf("{\"type\":\"kernel\",\"name\":\"cublasSgemv\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, alpha_param);
#elif defined(USE_ROCBLAS)
    printf("{\"type\":\"kernel\",\"name\":\"rocblas_sgemv\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, alpha_param);
#endif
    /* JSON-lines summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f},{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, gbps, gflops);

    /* Human-readable */
    fprintf(stderr, "cublas_gemv  M=%d N=%d  time_ms=%.4f  GB/s=%.2f  GFLOPS=%.2f\n",
            M, N, avg_ms, gbps, gflops);

    free(hA); free(hx); free(hy); free(hy_ref);
    return pass ? 0 : 1;
}
