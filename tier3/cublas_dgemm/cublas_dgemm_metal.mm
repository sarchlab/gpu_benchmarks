/**
 * cublas_dgemm_metal.mm — FP64 GEMM benchmark using Apple Accelerate cblas_dgemm.
 *
 * NOTE: Apple Metal does NOT support FP64 (double-precision) compute.
 *       This implementation uses cblas_dgemm from the Accelerate framework,
 *       which runs on the CPU (optimized with BLAS). No GPU is used.
 *
 * Computes: C = A * B
 *   where A is M×K, B is K×N, C is M×N (row-major, FP64)
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Foundation -framework Accelerate \
 *           cublas_dgemm_metal.mm -o cublas_dgemm
 */

#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

/* -------------------------------------------------------------------------
 * Benchmark parameters (defaults, overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int GEMM_M  = 4096;
static int GEMM_N  = 4096;
static int GEMM_K  = 4096;
static int WARMUP  = 0;
static int ITERS   = 5;
static double ALPHA = 1.0;

/* -------------------------------------------------------------------------
 * Timing helpers
 * ---------------------------------------------------------------------- */
static double ticks_to_ms(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-6;
}

/* -------------------------------------------------------------------------
 * CPU reference DGEMM — only for the small verification corner
 * A: vm×vk (row-major, full-width stride GEMM_K)
 * B: vk×vn (row-major, full-width stride GEMM_N)
 * C: vm×vn (row-major, stride vn)
 * ---------------------------------------------------------------------- */
static void cpu_dgemm_ref(int vm, int vn, int vk,
                          const double * __restrict__ A,
                          const double * __restrict__ B,
                          double       * __restrict__ C)
{
    for (int i = 0; i < vm; i++) {
        for (int j = 0; j < vn; j++) {
            double s = 0.0;
            for (int p = 0; p < vk; p++)
                s += A[(size_t)i * GEMM_K + p] * B[(size_t)p * GEMM_N + j];
            C[i * vn + j] = s;
        }
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    @autoreleasepool {

    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Read parameters from environment variables                          */
    /* ------------------------------------------------------------------ */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { GEMM_M = atoi(env_val); GEMM_N = GEMM_M; GEMM_K = GEMM_M; }
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) ITERS = atoi(env_val);
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) ALPHA = atof(env_val);

    const int M = GEMM_M;
    const int N = GEMM_N;
    const int K = GEMM_K;

    fprintf(stderr, "cublas_dgemm (Apple/Accelerate cblas_dgemm, CPU-only — Metal has no FP64)\n");
    fprintf(stderr, "M=%d  N=%d  K=%d  warmup=%d  iters=%d\n\n",
            M, N, K, WARMUP, ITERS);

    /* ------------------------------------------------------------------ */
    /* Allocate and initialize host matrices                               */
    /* ------------------------------------------------------------------ */
    double *A = (double *)malloc((size_t)M * K * sizeof(double));
    double *B = (double *)malloc((size_t)K * N * sizeof(double));
    double *C = (double *)malloc((size_t)M * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (double)rand() / (double)RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (double)rand() / (double)RAND_MAX;
    memset(C, 0, (size_t)M * N * sizeof(double));

    /* Helper: run one cblas_dgemm */
    auto run_dgemm = [&]() {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    ALPHA, A, K,
                         B, N,
                    0.0, C, N);
    };

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) run_dgemm();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    std::vector<double> times((size_t)ITERS);
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = mach_absolute_time();
        run_dgemm();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
        fprintf(stderr, "  iter %d: %.3f ms\n", i, times[i]);
    }

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs simple CPU DGEMM                        */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    std::vector<double> cpuC((size_t)VN * VN, 0.0);
    cpu_dgemm_ref(VN, VN, K, A, B, cpuC.data());

    /* cpuC is packed VN×VN (row stride = VN).
     * C is full M×N (row stride = N).
     * Compare element (row i, col j) from each. */
    bool pass = true;
    for (int i = 0; i < VN && pass; i++) {
        for (int j = 0; j < VN && pass; j++) {
            double ref = cpuC[i * VN + j];
            double got = C[(size_t)i * N + j];
            double err = fabs(ref - got) / (fabs(ref) + 1e-10);
            if (err > 1e-9) {
                fprintf(stderr, "MISMATCH at [%d,%d]: ref=%.15f got=%.15f rel=%.4e\n",
                        i, j, ref, got, err);
                pass = false;
            }
        }
    }
    fprintf(stderr, "Verification (first %d×%d corner): %s\n\n",
            VN, VN, pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Statistics                                                          */
    /* ------------------------------------------------------------------ */
    double sum_ms = 0.0;
    for (int i = 0; i < ITERS; i++) sum_ms += times[i];
    double avg_ms = sum_ms / ITERS;

    double gflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

    /* ------------------------------------------------------------------ */
    /* Report — JSON-lines to stdout                                       */
    /* ------------------------------------------------------------------ */
    printf("{\"type\":\"kernel\",\"name\":\"cublasDgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, ITERS, ALPHA);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           sum_ms, gflops);

    /* Human-readable to stderr */
    fprintf(stderr, "cublas_dgemm  M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f  (Accelerate cblas_dgemm, CPU-only)\n",
            M, N, K, avg_ms, gflops);

    free(A); free(B); free(C);
    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
