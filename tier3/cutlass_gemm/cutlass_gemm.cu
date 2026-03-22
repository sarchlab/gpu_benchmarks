/**
 * cutlass_gemm.cu — CUTLASS-style tiled FP32 GEMM benchmark.
 *
 * Custom tiled GEMM kernel with shared-memory staging and register tiling.
 * Self-contained — no external library dependencies.
 *
 * Computes: C = A * B
 *   where A is M×K, B is K×N, C is M×N (row-major, FP32)
 *
 * Tile parameters:
 *   Block tile:  BM×BN = 64×64
 *   K-tile:      BK = 8
 *   Thread tile: TM×TN = 8×8
 *   Threads/block: (BM/TM)*(BN/TN) = 8*8 = 64
 *
 * Algorithm:
 *   Each thread block computes a BM×BN tile of C.
 *   For each K-tile:
 *     1. Cooperatively load BM×BK of A and BK×BN of B into shared memory
 *     2. Each thread computes a TM×TN sub-tile using register tiling
 *        (outer-product accumulation over the BK dimension)
 *     3. Synchronize before the next K-tile
 *   Finally, each thread writes its TM×TN results to global memory.
 *
 * Verification: compare first 8×8 corner against CPU GEMM
 * GFLOPS = 2 * M * N * K / time_s / 1e9
 *
 * Output: JSON-lines protocol (stdout)
 *
 * Build:
 *   CUDA: nvcc -O3 -std=c++14 cutlass_gemm.cu -o cutlass_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Platform abstraction
 * ---------------------------------------------------------------------- */
#include <cuda_runtime.h>

#define GPU_CHECK(x) do {                                                  \
       cudaError_t _e = (x);                                                 \
       if (_e != cudaSuccess) {                                               \
           fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

typedef cudaEvent_t gpu_event_t;
#define GPU_EVENT_CREATE(e)        GPU_CHECK(cudaEventCreate(&(e)))
#define GPU_EVENT_RECORD(e)        GPU_CHECK(cudaEventRecord(e))
#define GPU_EVENT_SYNC(e)          GPU_CHECK(cudaEventSynchronize(e))
#define GPU_EVENT_ELAPSED(a,b,ms)  GPU_CHECK(cudaEventElapsedTime(&(ms),(a),(b)))
#define GPU_DEVICE_SYNC()          GPU_CHECK(cudaDeviceSynchronize())


/* -------------------------------------------------------------------------
 * Tile parameters
 * ---------------------------------------------------------------------- */
#define BM   64    /* Block tile rows */
#define BN   64    /* Block tile cols */
#define BK   8     /* K-tile size */
#define TM   8     /* Thread tile rows */
#define TN   8     /* Thread tile cols */

/* Number of threads per block: (BM/TM) * (BN/TN) = 8 * 8 = 64 */
#define NUM_THREADS  ((BM / TM) * (BN / TN))

/* -------------------------------------------------------------------------
 * Tiled GEMM kernel
 *
 * Each block computes a BM×BN tile of C.
 * Threads are arranged in an (BN/TN) × (BM/TM) = 8×8 grid.
 * Each thread computes a TM×TN = 8×8 sub-tile in registers.
 * ---------------------------------------------------------------------- */
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float*       __restrict__ C,
                                  int M, int N, int K)
{
    /* Shared memory for A and B tiles */
    __shared__ float smA[BM][BK];
    __shared__ float smB[BK][BN];

    /* Block position in the output grid */
    int bx = blockIdx.x;   /* column block index */
    int by = blockIdx.y;   /* row block index */

    /* Thread position within the 8×8 thread tile grid */
    int tx = threadIdx.x % (BN / TN);   /* 0..7 */
    int ty = threadIdx.x / (BN / TN);   /* 0..7 */

    /* Accumulator registers — each thread computes TM×TN outputs */
    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            regC[i][j] = 0.0f;

    /* Loop over K dimension in tiles of BK */
    for (int bk = 0; bk < K; bk += BK) {

        /* ---- Load A tile [BM × BK] into shared memory ---- */
        /* 512 elements / 64 threads = 8 elements per thread */
        #pragma unroll
        for (int load = 0; load < (BM * BK) / NUM_THREADS; load++) {
            int idx   = load * NUM_THREADS + threadIdx.x;
            int smRow = idx / BK;
            int smCol = idx % BK;
            int gRow  = by * BM + smRow;
            int gCol  = bk + smCol;
            smA[smRow][smCol] = (gRow < M && gCol < K)
                                ? A[(size_t)gRow * K + gCol]
                                : 0.0f;
        }

        /* ---- Load B tile [BK × BN] into shared memory ---- */
        /* 512 elements / 64 threads = 8 elements per thread */
        #pragma unroll
        for (int load = 0; load < (BK * BN) / NUM_THREADS; load++) {
            int idx   = load * NUM_THREADS + threadIdx.x;
            int smRow = idx / BN;
            int smCol = idx % BN;
            int gRow  = bk + smRow;
            int gCol  = bx * BN + smCol;
            smB[smRow][smCol] = (gRow < K && gCol < N)
                                ? B[(size_t)gRow * N + gCol]
                                : 0.0f;
        }

        __syncthreads();

        /* ---- Compute thread tile via outer-product accumulation ---- */
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            /* Load A column and B row into registers */
            float regA[TM], regB[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                regA[i] = smA[ty * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                regB[j] = smB[kk][tx * TN + j];

            /* Outer product */
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    regC[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    /* ---- Write results to global memory ---- */
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gRow = by * BM + ty * TM + i;
        if (gRow < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int gCol = bx * BN + tx * TN + j;
                if (gCol < N)
                    C[(size_t)gRow * N + gCol] = regC[i][j];
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * CPU reference GEMM — only for the small verification corner
 *
 * A: vm×vk (row-major, stride ldA)
 * B: vk×vn (row-major, stride ldB)
 * C: vm×vn (row-major, stride vn)
 * ---------------------------------------------------------------------- */
static void cpu_gemm(int vm, int vn, int vk,
                     const float * __restrict__ A, int ldA,
                     const float * __restrict__ B, int ldB,
                     float       * __restrict__ C)
{
    for (int i = 0; i < vm; i++) {
        for (int j = 0; j < vn; j++) {
            float s = 0.0f;
            for (int p = 0; p < vk; p++)
                s += A[(size_t)i * ldA + p] * B[(size_t)p * ldB + j];
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

    int M = 4096;
    int N = 4096;
    int K = 4096;
    int warmup = 5;
    int iters = 5;
    int block_tile_m = BM;

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; K = M; }
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) warmup = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) iters = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_tile_m");
    if (env_val) block_tile_m = atoi(env_val);

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
    for (size_t i = 0; i < (size_t)M * K; i++)
        hA[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; i++)
        hB[i] = (float)rand() / (float)RAND_MAX;
    memset(hC, 0, (size_t)M * N * sizeof(float));

    fprintf(stderr, "cutlass_gemm: M=%d N=%d K=%d  BM=%d BN=%d BK=%d TM=%d TN=%d\n",
            M, N, K, BM, BN, BK, TM, TN);
    fprintf(stderr, "  threads/block=%d  grid=(%d, %d)\n\n",
            NUM_THREADS, N / BN, M / BM);

    /* ------------------------------------------------------------------ */
    /* Device memory                                                       */
    /* ------------------------------------------------------------------ */
    float *dA, *dB, *dC;

    GPU_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dA, hA, (size_t)M * K * sizeof(float),
                         cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, hB, (size_t)K * N * sizeof(float),
                         cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

    /* ------------------------------------------------------------------ */
    /* Kernel launch configuration                                         */
    /* ------------------------------------------------------------------ */
    dim3 block(NUM_THREADS);              /* 64 threads */
    dim3 grid(N / BN, M / BM);           /* (64, 64)   */

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < warmup; i++) {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    GPU_DEVICE_SYNC();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    gpu_event_t ev_start, ev_stop;
    GPU_EVENT_CREATE(ev_start);
    GPU_EVENT_CREATE(ev_stop);

    GPU_EVENT_RECORD(ev_start);
    for (int i = 0; i < iters; i++) {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)iters;

    /* ------------------------------------------------------------------ */
    /* Copy result back                                                    */
    /* ------------------------------------------------------------------ */
    GPU_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU GEMM                                */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    float *cpuC = (float *)calloc((size_t)VN * VN, sizeof(float));
    cpu_gemm(VN, VN, K, hA, K, hB, N, cpuC);

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
    /* Report — JSON-lines protocol                                        */
    /* ------------------------------------------------------------------ */
    double gflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

    /* Kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"tiled_gemm_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"block_tile_m\":%d}}\n",
           (double)avg_ms, M, warmup, iters, block_tile_m);

    /* Summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, gflops);

    fprintf(stderr, "cutlass_gemm  M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            M, N, K, avg_ms, gflops);

    free(hA); free(hB); free(hC);
    return pass ? 0 : 1;
}
