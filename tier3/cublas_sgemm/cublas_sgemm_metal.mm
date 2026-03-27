/**
 * cublas_sgemm_metal.mm — FP32 GEMM benchmark using Apple Metal Performance Shaders.
 *
 * Computes: C = A * B
 *   where A is M×K, B is K×N, C is M×N (row-major, FP32)
 *
 * Uses MPSMatrixMultiplication (part of Metal Performance Shaders).
 * Matrices are stored row-major; MPSMatrixDescriptor rowBytes = cols * sizeof(float).
 *
 * Parameters read from BENCH_PARAM_* environment variables.
 * Output: JSON-lines to stdout, human-readable info to stderr.
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework MetalPerformanceShaders \
 *           -framework Foundation -framework Accelerate \
 *           cublas_sgemm_metal.mm -o cublas_sgemm
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
 * Benchmark parameters (defaults, overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int GEMM_M  = 4096;
static int GEMM_N  = 4096;
static int GEMM_K  = 4096;
static int WARMUP  = 0;
static float ALPHA  = 1.0f;

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
 * CPU reference GEMM — only for the small verification corner
 * A: vm×vk (row-major, full-width stride GEMM_K)
 * B: vk×vn (row-major, full-width stride GEMM_N)
 * C: vm×vn (row-major, stride vn)
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
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) ALPHA = (float)atof(env_val);

    const int M = GEMM_M;
    const int N = GEMM_N;
    const int K = GEMM_K;

    /* ------------------------------------------------------------------ */
    /* Metal device setup                                                  */
    /* ------------------------------------------------------------------ */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return 1;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "M=%d  N=%d  K=%d  warmup=%d\n\n",
            M, N, K, WARMUP);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------ */
    /* Allocate Metal buffers (Shared storage — CPU-writable, GPU-readable)*/
    /* ------------------------------------------------------------------ */
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

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
    float *hA = (float *)bufA.contents;
    float *hB = (float *)bufB.contents;
    float *hC = (float *)bufC.contents;

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / (float)RAND_MAX;
    memset(hC, 0, bytesC);

    /* ------------------------------------------------------------------ */
    /* Create MPSMatrix objects                                            */
    /*                                                                     */
    /* Row-major layout: rowBytes = cols * sizeof(float)                  */
    /* MPS uses row-major by default for MPSMatrix.                       */
    /* ------------------------------------------------------------------ */
    NSUInteger rowBytesA = (NSUInteger)K * sizeof(float);
    NSUInteger rowBytesB = (NSUInteger)N * sizeof(float);
    NSUInteger rowBytesC = (NSUInteger)N * sizeof(float);

    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)K
                                             rowBytes:rowBytesA
                                            dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)K
                                              columns:(NSUInteger)N
                                             rowBytes:rowBytesB
                                            dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)N
                                             rowBytes:rowBytesC
                                            dataType:MPSDataTypeFloat32];

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
                                              alpha:(double)ALPHA
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
    uint64_t t0 = mach_absolute_time();
    run_gemm();
    uint64_t t1 = mach_absolute_time();
    double total_ms = ticks_to_ms(t1 - t0);

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU SGEMM                               */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    std::vector<float> cpuC((size_t)VN * VN, 0.0f);
    cpu_sgemm(VN, VN, K, hA, hB, cpuC.data());

    /* cpuC is packed VN×VN (row stride = VN).
     * hC is full M×N (row stride = N).
     * Compare element (row i, col j) from each. */
    bool pass = true;
    for (int i = 0; i < VN && pass; i++) {
        for (int j = 0; j < VN && pass; j++) {
            float ref = cpuC[i * VN + j];
            float got = hC[(size_t)i * N + j];
            float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
            if (err > 1e-3f) {
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
    double avg_ms = total_ms;

    double gflops = 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

    /* ------------------------------------------------------------------ */
    /* Report — JSON-lines to stdout                                       */
    /* ------------------------------------------------------------------ */
    printf("{\"type\":\"kernel\",\"name\":\"cublasSgemm\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, (double)ALPHA);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           total_ms, gflops);

    /* Human-readable to stderr */
    fprintf(stderr, "cublas_sgemm  M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            M, N, K, avg_ms, gflops);

    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
