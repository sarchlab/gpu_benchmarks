/**
 * cublas_hgemm_metal.mm — FP16 GEMM benchmark using Apple Metal Performance Shaders.
 *
 * Computes: C = alpha * A * B + beta * C
 *   where A is M×K, B is K×N, C is M×N (row-major, FP16)
 *   M = N = K = 4096, alpha = 1.0, beta = 0.0
 *
 * Uses MPSMatrixMultiplication (part of Metal Performance Shaders).
 * Matrices are stored row-major; MPSMatrixDescriptor rowBytes = cols * sizeof(__fp16).
 *
 * Warmup: 5 command buffers (not timed)
 * Timed:  5 command buffers, report average via mach_absolute_time()
 *
 * Verification: compare first 8×8 corner of C against CPU GEMM
 * TFLOPS = 2 * M * N * K / time_s / 1e12
 *
 * Output:
 *   CSV  (stdout): cublas_hgemm,<M>,<time_ms>,<TFLOPS>
 *   Info (stderr): device name, per-iteration times, verification result
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework MetalPerformanceShaders \
 *           -framework Foundation -framework Accelerate \
 *           cublas_hgemm_metal.mm -o cublas_hgemm
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <mach/mach_time.h>

/* -------------------------------------------------------------------------
 * Benchmark default parameters (overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */

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
 * CPU reference GEMM — converts FP16 inputs to float for accumulation,
 * only for the small verification corner
 * A: vm×vk (row-major, full-width stride GEMM_K)
 * B: vk×vn (row-major, full-width stride GEMM_N)
 * C: vm×vn (row-major, stride vn) — float output
 * ---------------------------------------------------------------------- */
static void cpu_hgemm_ref(int vm, int vn, int vk,
                           const __fp16 * __restrict__ A, int lda,
                           const __fp16 * __restrict__ B, int ldb,
                           float         * __restrict__ C)
{
    for (int i = 0; i < vm; i++) {
        for (int j = 0; j < vn; j++) {
            float s = 0.0f;
            for (int p = 0; p < vk; p++)
                s += (float)A[(size_t)i * lda + p] *
                     (float)B[(size_t)p * ldb + j];
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
    /* Benchmark parameters (overridable via BENCH_PARAM_* env vars)       */
    /* ------------------------------------------------------------------ */
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int WARMUP = 5;
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
    /* Metal device setup                                                  */
    /* ------------------------------------------------------------------ */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return 1;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "M=%d  N=%d  K=%d  warmup=%d  iters=%d\n\n",
            M, N, K, WARMUP, ITERS);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------ */
    /* Allocate Metal buffers (Shared storage — CPU-writable, GPU-readable)*/
    /* ------------------------------------------------------------------ */
    size_t bytesA = (size_t)M * K * sizeof(__fp16);
    size_t bytesB = (size_t)K * N * sizeof(__fp16);
    size_t bytesC = (size_t)M * N * sizeof(__fp16);

    id<MTLBuffer> bufA = [device newBufferWithLength:bytesA
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithLength:bytesB
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytesC
                                             options:MTLResourceStorageModeShared];

    if (!bufA || !bufB || !bufC) {
        fprintf(stderr, "Error: failed to allocate Metal buffers\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Initialize input matrices                                           */
    /* ------------------------------------------------------------------ */
    __fp16 *hA = (__fp16 *)bufA.contents;
    __fp16 *hB = (__fp16 *)bufB.contents;
    __fp16 *hC = (__fp16 *)bufC.contents;

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (__fp16)((float)rand() / (float)RAND_MAX);
    for (int i = 0; i < K * N; i++) hB[i] = (__fp16)((float)rand() / (float)RAND_MAX);
    memset(hC, 0, bytesC);

    /* ------------------------------------------------------------------ */
    /* Create MPSMatrix objects                                            */
    /*                                                                     */
    /* Row-major layout: rowBytes = cols * sizeof(__fp16)                  */
    /* MPS uses row-major by default for MPSMatrix.                       */
    /* ------------------------------------------------------------------ */
    NSUInteger rowBytesA = (NSUInteger)K * sizeof(__fp16);
    NSUInteger rowBytesB = (NSUInteger)N * sizeof(__fp16);
    NSUInteger rowBytesC = (NSUInteger)N * sizeof(__fp16);

    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)K
                                             rowBytes:rowBytesA
                                            dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)K
                                              columns:(NSUInteger)N
                                             rowBytes:rowBytesB
                                            dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)N
                                             rowBytes:rowBytesC
                                            dataType:MPSDataTypeFloat16];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    /* ------------------------------------------------------------------ */
    /* Create MPSMatrixMultiplication kernel                               */
    /* C = alpha * A * B + beta * C                                       */
    /* ------------------------------------------------------------------ */
    MPSMatrixMultiplication *matMul =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                      transposeLeft:NO
                                     transposeRight:NO
                                         resultRows:(NSUInteger)M
                                      resultColumns:(NSUInteger)N
                                    interiorColumns:(NSUInteger)K
                                              alpha:(double)alpha_param
                                               beta:0.0];

    /* Helper lambda: encode + commit one GEMM */
    auto run_gemm = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [matMul encodeToCommandBuffer:cb
                           leftMatrix:matA
                          rightMatrix:matB
                        resultMatrix:matC];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) run_gemm();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    std::vector<double> times((size_t)ITERS);
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = mach_absolute_time();
        run_gemm();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
        fprintf(stderr, "  iter %d: %.3f ms\n", i, times[i]);
    }

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU reference                            */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    std::vector<float> cpuC((size_t)VN * VN, 0.0f);
    cpu_hgemm_ref(VN, VN, K, hA, K, hB, N, cpuC.data());

    /* cpuC is packed VN×VN (row stride = VN), float.
     * hC is full M×N (row stride = N), __fp16.
     * Compare element (row i, col j) from each. */
    bool pass = true;
    for (int i = 0; i < VN && pass; i++) {
        for (int j = 0; j < VN && pass; j++) {
            float ref = cpuC[i * VN + j];
            float got = (float)hC[(size_t)i * N + j];
            float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
            if (err > 5e-2f) {
                fprintf(stderr, "MISMATCH at [%d,%d]: ref=%.6f got=%.6f rel=%.4e\n",
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

    double tflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e12;

    /* JSON-lines kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"cublasHgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, ITERS, alpha_param);
    /* JSON-lines summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"TFLOPS\",\"value\":%.2f}]}\n",
           sum_ms, tflops);

    /* Human-readable */
    fprintf(stderr, "cublas_hgemm  M=%d N=%d K=%d  time_ms=%.3f  TFLOPS=%.2f\n",
            M, N, K, avg_ms, tflops);

    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
