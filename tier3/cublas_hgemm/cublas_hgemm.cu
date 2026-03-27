/**
 * cublas_hgemm.cu — FP16 GEMM benchmark using cuBLAS (NVIDIA).
 *
 * Computes: C = alpha * A * B + beta * C
 *   where A is M×K, B is K×N, C is M×N (all row-major, FP16)
 *   M = N = K = 4096, alpha = 1.0, beta = 0.0
 *
 * Row-major GEMM via col-major BLAS trick:
 *   Row-major C = A*B  ↔  Col-major C^T = B^T * A^T
 *   cublasHgemm with op_none, op_none, dims (N, M, K)
 *   and leading dimensions: ldB=N, ldA=K, ldC=N
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations, report average
 *
 * Verification: compare first 8×8 corner of C against CPU GEMM
 * TFLOPS = 2 * M * N * K / time_s / 1e12
 *
 * Output:
 *   cublas_hgemm,<M>,<time_ms>,<TFLOPS>
 *
 * Build:
 *   CUDA:   nvcc -O3 -std=c++14 cublas_hgemm.cu -lcublas -o cublas_hgemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define GPU_CHECK(x) do {                                                  \
       cudaError_t _e = (x);                                                 \
       if (_e != cudaSuccess) {                                               \
           fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#define BLAS_CHECK(x) do {                                                 \
       cublasStatus_t _e = (x);                                              \
       if (_e != CUBLAS_STATUS_SUCCESS) {                                     \
           fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

typedef cudaEvent_t gpu_event_t;
#define GPU_EVENT_CREATE(e)     GPU_CHECK(cudaEventCreate(&(e)))
#define GPU_EVENT_RECORD(e)     GPU_CHECK(cudaEventRecord(e))
#define GPU_EVENT_SYNC(e)       GPU_CHECK(cudaEventSynchronize(e))
#define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(cudaEventElapsedTime(&(ms),(a),(b)))
#define GPU_DEVICE_SYNC()       GPU_CHECK(cudaDeviceSynchronize())

typedef __half half_t;
static inline half_t to_half(float f) { return __float2half(f); }
static inline float  to_float(half_t h) { return __half2float(h); }


/* -------------------------------------------------------------------------
 * Benchmark default parameters (overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * CPU reference GEMM — converts FP16 inputs to float for accumulation,
 * only for a small corner — O(verify_m * verify_n * K)
 * ---------------------------------------------------------------------- */
static void cpu_hgemm_ref(int vm, int vn, int vk,
                           const half_t * __restrict__ A, int lda,
                           const half_t * __restrict__ B, int ldb,
                           float        * __restrict__ C)
{
    /* A is vm×vk (row-major, stride lda)
     * B is vk×vn (row-major, stride ldb)
     * C is vm×vn (row-major, stride vn) — float output */
    for (int i = 0; i < vm; i++) {
        for (int j = 0; j < vn; j++) {
            float s = 0.0f;
            for (int p = 0; p < vk; p++)
                s += to_float(A[(size_t)i * lda + p]) *
                     to_float(B[(size_t)p * ldb + j]);
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
    /* Benchmark parameters (overridable via BENCH_PARAM_* env vars)       */
    /* ------------------------------------------------------------------ */
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int WARMUP = 0;
    int ITERS = 5;
    float alpha_param = 1.0f;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; K = M; }
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) ITERS = atoi(env_val);
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha_param = (float)atof(env_val);

    /* ------------------------------------------------------------------ */
    /* Allocate and initialize host matrices in float, then convert       */
    /* ------------------------------------------------------------------ */
    float *hA_f = (float *)malloc((size_t)M * K * sizeof(float));
    float *hB_f = (float *)malloc((size_t)K * N * sizeof(float));

    if (!hA_f || !hB_f) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) hA_f[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) hB_f[i] = (float)rand() / (float)RAND_MAX;

    /* Convert to FP16 */
    half_t *hA = (half_t *)malloc((size_t)M * K * sizeof(half_t));
    half_t *hB = (half_t *)malloc((size_t)K * N * sizeof(half_t));
    half_t *hC = (half_t *)malloc((size_t)M * N * sizeof(half_t));

    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < M * K; i++) hA[i] = to_half(hA_f[i]);
    for (int i = 0; i < K * N; i++) hB[i] = to_half(hB_f[i]);
    memset(hC, 0, (size_t)M * N * sizeof(half_t));

    free(hA_f);
    free(hB_f);

    /* ------------------------------------------------------------------ */
    /* Device memory                                                        */
    /* ------------------------------------------------------------------ */
    half_t *dA, *dB, *dC;

    GPU_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(half_t)));
    GPU_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(half_t)));
    GPU_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(half_t)));
    GPU_CHECK(cudaMemcpy(dA, hA, (size_t)M * K * sizeof(half_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, hB, (size_t)K * N * sizeof(half_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(half_t)));

    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));

    half_t alpha = to_half(alpha_param), beta = to_half(0.0f);

#define RUN_HGEMM() \
        BLAS_CHECK(cublasHgemm(handle,                              \
            CUBLAS_OP_N, CUBLAS_OP_N,                               \
            N, M, K,                                                \
            &alpha, dB, N, dA, K, &beta, dC, N))


    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) {
        RUN_HGEMM();
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
        RUN_HGEMM();
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)ITERS;

    /* ------------------------------------------------------------------ */
    /* Copy result back                                                    */
    /* ------------------------------------------------------------------ */
    GPU_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(half_t), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU reference                            */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    float *cpuC = (float *)calloc((size_t)VN * VN, sizeof(float));
    cpu_hgemm_ref(VN, VN, K, hA, K, hB, N, cpuC);

    /* cpuC is packed VN×VN (row stride = VN), float.
     * hC is full M×N (row stride = N), half_t.
     * Compare element (row i, col j) from each. */
    int pass = 1;
    for (int i = 0; i < VN && pass; i++) {
        for (int j = 0; j < VN && pass; j++) {
            float ref = cpuC[i * VN + j];
            float got = to_float(hC[(size_t)i * N + j]);
            float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
            if (err > 5e-2f) {
                fprintf(stderr, "MISMATCH at [%d,%d]: ref=%.6f got=%.6f rel=%.4e\n",
                        i, j, ref, got, err);
                pass = 0;
            }
        }
    }
    free(cpuC);
    fprintf(stderr, "Verification (first %d×%d corner): %s\n", VN, VN,
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    double tflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e12;

    /* JSON-lines kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"cublasHgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, ITERS, alpha_param);
    /* JSON-lines summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"TFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, tflops);

    /* Human-readable */
    fprintf(stderr, "cublas_hgemm  M=%d N=%d K=%d  time_ms=%.3f  TFLOPS=%.2f\n",
            M, N, K, avg_ms, tflops);

    free(hA); free(hB); free(hC);
    return pass ? 0 : 1;
}
