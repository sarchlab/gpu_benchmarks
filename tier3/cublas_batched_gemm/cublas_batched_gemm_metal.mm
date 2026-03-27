/**
 * cublas_batched_gemm_metal.mm — Batched FP32 GEMM using Apple Metal Performance Shaders.
 *
 * Computes: C[i] = alpha * A[i] * B[i] + beta * C[i]  for i in [0, batch_count)
 *   where each A[i] is M×K, B[i] is K×N, C[i] is M×N (row-major, FP32)
 *   batch_count = 64, M = N = K = 512, alpha = 1.0, beta = 0.0
 *
 * Uses MPSMatrixMultiplication (part of Metal Performance Shaders).
 * All batch_count multiplications are encoded into a single command buffer.
 * Matrices are stored row-major; MPSMatrixDescriptor rowBytes = cols * sizeof(float).
 *
 * Warmup: 5 command buffers (not timed)
 * Timed:  5 command buffers, report average via mach_absolute_time()
 *
 * Verification: compare first 8×8 corner of C[0] against CPU GEMM
 * GFLOPS = batch_count * 2 * M * N * K / time_s / 1e9
 *
 * Output:
 *   CSV  (stdout): cublas_batched_gemm,64x512,<time_ms>,<GFLOPS>
 *   Info (stderr): device name, per-iteration times, verification result
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework MetalPerformanceShaders \
 *           -framework Foundation -framework Accelerate \
 *           cublas_batched_gemm_metal.mm -o cublas_batched_gemm
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
 * Benchmark parameters (runtime-configurable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int BATCH_COUNT = 64;
static int GEMM_M      = 512;
static int GEMM_N      = 512;
static int GEMM_K      = 512;
static int WARMUP      = 0;
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

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { GEMM_M = atoi(env_val); GEMM_N = GEMM_M; GEMM_K = GEMM_M; }
    env_val = getenv("BENCH_PARAM_batch_count");
    if (env_val) BATCH_COUNT = atoi(env_val);
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    const int M = GEMM_M;
    const int N = GEMM_N;
    const int K = GEMM_K;
    const size_t sizeA = (size_t)M * K;
    const size_t sizeB = (size_t)K * N;
    const size_t sizeC = (size_t)M * N;

    /* ------------------------------------------------------------------ */
    /* Metal device setup                                                  */
    /* ------------------------------------------------------------------ */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return 1;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "batch_count=%d  M=%d  N=%d  K=%d  warmup=%d\n\n",
            BATCH_COUNT, M, N, K, WARMUP);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------ */
    /* Allocate Metal buffers — one large buffer per matrix type           */
    /* Each batch element is contiguous within the buffer.                 */
    /* ------------------------------------------------------------------ */
    size_t bytesA = BATCH_COUNT * sizeA * sizeof(float);
    size_t bytesB = BATCH_COUNT * sizeB * sizeof(float);
    size_t bytesC = BATCH_COUNT * sizeC * sizeof(float);

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
    for (size_t i = 0; i < BATCH_COUNT * sizeA; i++)
        hA[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < BATCH_COUNT * sizeB; i++)
        hB[i] = (float)rand() / (float)RAND_MAX;
    memset(hC, 0, bytesC);

    /* ------------------------------------------------------------------ */
    /* Create per-batch MPSMatrix objects                                  */
    /*                                                                     */
    /* Each MPSMatrix wraps a region of the shared buffer at an offset.    */
    /* Row-major layout: rowBytes = cols * sizeof(float)                  */
    /* ------------------------------------------------------------------ */
    NSUInteger rowBytesA = (NSUInteger)K * sizeof(float);
    NSUInteger rowBytesB = (NSUInteger)N * sizeof(float);
    NSUInteger rowBytesC = (NSUInteger)N * sizeof(float);

    NSMutableArray<MPSMatrix *> *matsA = [NSMutableArray arrayWithCapacity:BATCH_COUNT];
    NSMutableArray<MPSMatrix *> *matsB = [NSMutableArray arrayWithCapacity:BATCH_COUNT];
    NSMutableArray<MPSMatrix *> *matsC = [NSMutableArray arrayWithCapacity:BATCH_COUNT];

    for (int b = 0; b < BATCH_COUNT; b++) {
        NSUInteger offsetA = b * sizeA * sizeof(float);
        NSUInteger offsetB = b * sizeB * sizeof(float);
        NSUInteger offsetC = b * sizeC * sizeof(float);

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

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA
                                                     offset:offsetA
                                                 descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB
                                                     offset:offsetB
                                                 descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC
                                                     offset:offsetC
                                                 descriptor:descC];

        [matsA addObject:matA];
        [matsB addObject:matB];
        [matsC addObject:matC];
    }

    /* ------------------------------------------------------------------ */
    /* Create MPSMatrixMultiplication kernel                               */
    /* C[i] = alpha * A[i] * B[i] + beta * C[i]                          */
    /* ------------------------------------------------------------------ */
    MPSMatrixMultiplication *matMul =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                      transposeLeft:NO
                                     transposeRight:NO
                                         resultRows:(NSUInteger)M
                                      resultColumns:(NSUInteger)N
                                    interiorColumns:(NSUInteger)K
                                              alpha:1.0
                                               beta:0.0];

    /* Helper lambda: encode all batch_count GEMMs into one command buffer */
    auto run_batched_gemm = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int b = 0; b < BATCH_COUNT; b++) {
            [matMul encodeToCommandBuffer:cb
                               leftMatrix:matsA[b]
                              rightMatrix:matsB[b]
                            resultMatrix:matsC[b]];
        }
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) run_batched_gemm();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
        uint64_t t0 = mach_absolute_time();
    run_batched_gemm();
    uint64_t t1 = mach_absolute_time();
    double total_ms = ticks_to_ms(t1 - t0);

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner of C[0] vs CPU SGEMM                       */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    std::vector<float> cpuC((size_t)VN * VN, 0.0f);
    cpu_sgemm(VN, VN, K, hA, hB, cpuC.data());

    /* cpuC is packed VN×VN (row stride = VN).
     * hC is first batch C[0], full M×N (row stride = N). */
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
    fprintf(stderr, "Verification (first %d×%d corner of batch 0): %s\n\n",
            VN, VN, pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Statistics                                                          */
    /* ------------------------------------------------------------------ */
    double avg_ms = total_ms;

    double gflops = (double)BATCH_COUNT * 2.0 * M * N * K / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines output */
    printf("{\"type\":\"kernel\",\"name\":\"cublasSgemmBatched\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"batch_count\":%d,\"warmup\":%d}}\n",
           avg_ms, M, BATCH_COUNT, WARMUP);
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           total_ms, gflops);

    /* Human-readable */
    fprintf(stderr, "cublas_batched_gemm  batch=%d M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            BATCH_COUNT, M, N, K, avg_ms, gflops);

    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
