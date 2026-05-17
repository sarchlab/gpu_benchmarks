/**
 * cutlass_gemm_metal.mm — Metal host code for CUTLASS-style tiled FP32 GEMM.
 *
 * Dispatches a custom Metal compute shader (cutlass_gemm.metal) that implements
 * tiled GEMM with threadgroup-memory staging and register tiling.
 * Self-contained — no vendor library dependencies (no MPS).
 *
 * Computes: C = A * B
 *   where A is M×K, B is K×N, C is M×N (row-major, FP32)
 *
 * Verification: compare first 8×8 corner against CPU GEMM
 * GFLOPS = 2 * M * N * K / time_s / 1e9
 *
 * Output: JSON-lines protocol (stdout)
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework Foundation \
 *           cutlass_gemm_metal.mm -o cutlass_gemm
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <mach/mach_time.h>

/* Tile parameters — must match the Metal shader */
#define BM   64
#define BN   64
#define TM   8
#define TN   8
#define NUM_THREADS  ((BM / TM) * (BN / TN))   /* 64 */

/* Parameter struct — must match Metal shader layout */
struct GemmParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

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
int main(int argc, char *argv[])
{
    @autoreleasepool {

    setvbuf(stdout, NULL, _IONBF, 0);

    int M = 4096;
    int N = 4096;
    int K = 4096;
    int warmup = 0;
    int block_tile_m = BM;

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; K = M; }
    env_val = getenv("BENCH_PARAM_block_tile_m");
    if (env_val) block_tile_m = atoi(env_val);

    /* ------------------------------------------------------------------ */
    /* Metal device setup                                                  */
    /* ------------------------------------------------------------------ */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return 1;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "M=%d  N=%d  K=%d  BM=%d BN=%d TM=%d TN=%d\n",
            M, N, K, BM, BN, TM, TN);
    fprintf(stderr, "threads/group=%d  warmup=%d\n\n",
            NUM_THREADS, warmup);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------ */
    /* Compile Metal shader at runtime                                     */
    /* ------------------------------------------------------------------ */
    NSError *err = nil;
    id<MTLLibrary> library = nil;
    {
        /* Look for cutlass_gemm.metal next to the binary */
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"cutlass_gemm.metal"];
        NSString *metalSrc = nil;

        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:nil];
        }

        if (metalSrc) {
            MTLCompileOptions *opts = [MTLCompileOptions new];
            library = [device newLibraryWithSource:metalSrc
                                           options:opts
                                             error:&err];
            if (!library) {
                fprintf(stderr, "Metal compile error: %s\n",
                        err.localizedDescription.UTF8String);
                return 1;
            }
        } else {
            fprintf(stderr,
                    "Error: cutlass_gemm.metal not found next to binary at %s\n",
                    srcPath.UTF8String);
            return 1;
        }
    }

    id<MTLFunction> kernelFunc = [library newFunctionWithName:@"tiled_gemm"];
    if (!kernelFunc) {
        fprintf(stderr, "Error: kernel function 'tiled_gemm' not found\n");
        return 1;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:kernelFunc error:&err];
    if (!pipeline) {
        fprintf(stderr, "Error: failed to create pipeline state: %s\n",
                err.localizedDescription.UTF8String);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Allocate Metal buffers (Shared storage — CPU+GPU accessible)        */
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
    for (size_t i = 0; i < (size_t)M * K; i++)
        hA[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; i++)
        hB[i] = (float)rand() / (float)RAND_MAX;
    memset(hC, 0, bytesC);

    /* ------------------------------------------------------------------ */
    /* Create params buffer                                                */
    /* ------------------------------------------------------------------ */
    GemmParams params = { (uint32_t)M, (uint32_t)N, (uint32_t)K };
    id<MTLBuffer> bufParams = [device newBufferWithBytes:&params
                                                  length:sizeof(GemmParams)
                                                 options:MTLResourceStorageModeShared];

    /* ------------------------------------------------------------------ */
    /* Threadgroup and grid sizes                                          */
    /* ------------------------------------------------------------------ */
    MTLSize threadgroupSize = MTLSizeMake(NUM_THREADS, 1, 1);
    MTLSize gridSize = MTLSizeMake((N + BN - 1) / BN,
                                   (M + BM - 1) / BM, 1);

    /* Helper: encode + commit + wait for one GEMM dispatch */
    auto run_gemm = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA      offset:0 atIndex:0];
        [enc setBuffer:bufB      offset:0 atIndex:1];
        [enc setBuffer:bufC      offset:0 atIndex:2];
        [enc setBuffer:bufParams offset:0 atIndex:3];
        [enc dispatchThreadgroups:gridSize
           threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < warmup; i++) run_gemm();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
        uint64_t t0 = mach_absolute_time();
    run_gemm();
    uint64_t t1 = mach_absolute_time();
    double total_ms = ticks_to_ms(t1 - t0);

    /* ------------------------------------------------------------------ */
    /* Verify first 8×8 corner vs CPU GEMM                                */
    /* ------------------------------------------------------------------ */
    const int VN = 8;
    std::vector<float> cpuC((size_t)VN * VN, 0.0f);
    cpu_gemm(VN, VN, K, hA, K, hB, N, cpuC.data());

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
    /* Report — JSON-lines protocol                                        */
    /* ------------------------------------------------------------------ */

    /* Kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"tiled_gemm_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"block_tile_m\":%d}}\n",
           avg_ms, M, warmup, block_tile_m);

    /* Summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           total_ms, gflops);

    fprintf(stderr, "cutlass_gemm  M=%d N=%d K=%d  time_ms=%.3f  GFLOPS=%.2f\n",
            M, N, K, avg_ms, gflops);

    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
