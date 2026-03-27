/**
 * cublas_sgemm.cu — FP32 GEMM benchmark using cuBLAS (NVIDIA).
 *
 * Computes: C = A * B
 *   where A is M×K, B is K×N, C is M×N (all row-major, FP32)
 *
 * Row-major GEMM via col-major BLAS trick:
 *   Row-major C = A*B  ↔  Col-major C^T = B^T * A^T
 *   cublasSgemm with op_none, op_none, dims (N, M, K)
 *   and leading dimensions: ldB=N, ldA=K, ldC=N
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   CUDA:   nvcc -O3 -std=c++14 cublas_sgemm.cu -lcublas -o cublas_sgemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

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


/* -------------------------------------------------------------------------
 * Benchmark parameters (defaults, overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int M      = 4096;
static int N      = 4096;
static int K      = 4096;
static int WARMUP = 0;
static int ITERS  = 5;

/* -------------------------------------------------------------------------
 * CPU reference GEMM (only for a small corner — O(verify_m * verify_n * K))
 * ---------------------------------------------------------------------- */
static void cpu_sgemm(int vm, int vn, int vk,
                      const float * __restrict__ A,
                      const float * __restrict__ B,
                      float       * __restrict__ C)
{
    /* A is vm×vk (row-major, stride K)
     * B is vk×vn (row-major, stride N)
     * C is vm×vn (row-major, stride vn) */
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
    float *hA = (float *)malloc((size_t)M * K * sizeof(float));
    float *hB = (float *)malloc((size_t)K * N * sizeof(float));
    float *hC = (float *)malloc((size_t)M * N * sizeof(float));

    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / (float)RAND_MAX;
    memset(hC, 0, (size_t)M * N * sizeof(float));

    /* ------------------------------------------------------------------ */
    /* Device memory                                                        */
    /* ------------------------------------------------------------------ */
    float *dA, *dB, *dC;
    float alpha = 1.0f, beta = 0.0f;

    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha = (float)atof(env_val);

    GPU_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dA, hA, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, hB, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

    cublasHandle_t handle;
    BLAS_CHECK(cublasCreate(&handle));

#define RUN_SGEMM() \
        BLAS_CHECK(cublasSgemm(handle,                              \
            CUBLAS_OP_N, CUBLAS_OP_N,                               \
            N, M, K,                                                \
            &alpha, dB, N, dA, K, &beta, dC, N))


    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) {
        RUN_SGEMM();
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
        RUN_SGEMM();
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)ITERS;

    /* ------------------------------------------------------------------ */
    /* Copy result back                                                    */
    /* ------------------------------------------------------------------ */
    GPU_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU SGEMM                               */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    float *cpuC = (float *)calloc((size_t)VN * VN, sizeof(float));
    cpu_sgemm(VN, VN, K, hA, hB, cpuC);

    /* cpuC is packed VN×VN (row stride = VN).
     * hC is full M×N (row stride = N).
     * Compare element (row i, col j) from each. */
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
    fprintf(stderr, "Verification (first %d×%d corner): %s\n", VN, VN,
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Report — JSON-lines to stdout                                       */
    /* ------------------------------------------------------------------ */
    double gflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

    printf("{\"type\":\"kernel\",\"name\":\"cublasSgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           (double)avg_ms, M, WARMUP, ITERS, (double)alpha);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, gflops);

    /* Human-readable to stderr */
    fprintf(stderr, "cublas_sgemm  M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            M, N, K, (double)avg_ms, gflops);

    free(hA); free(hB); free(hC);
    return pass ? 0 : 1;
}
