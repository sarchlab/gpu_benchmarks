/**
 * tensor_core_throughput_metal.mm
 *
 * Apple Metal host code for measuring simdgroup matrix multiply throughput
 * (equivalent to Tensor Cores on NVIDIA/AMD).
 *
 * Uses simdgroup_matrix_multiply_accumulate in an 8×8 tiled GEMM to measure
 * peak half-precision matrix multiply performance.
 *
 * Matrix size:  M = N = K = 4096 (default)
 * Timing:       warmup + timed iterations, report average
 *
 * Usage:
 *   ./tensor_core_throughput [matrix_dim]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_matrix_dim
 *   BENCH_PARAM_precision          ("fp16" or "all")
 *   BENCH_PARAM_warmup_iterations
 *   BENCH_PARAM_timed_iterations
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_DIM               = 4096;
static const int SIMD_TILE                 = 8;    // 8×8 simdgroup tiles
static const int SIMDGROUP_SIZE            = 32;   // threads per simdgroup
static const int DEFAULT_WARMUP_ITERATIONS = 5;
static const int DEFAULT_TIMED_ITERATIONS  = 5;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double ticks_to_ms() {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)info.numer / (double)info.denom / 1e6;
}

// ---------------------------------------------------------------------------
// Metal shader source (embedded)
// ---------------------------------------------------------------------------

static const char *metalShaderSource = R"(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_gemm(device const half *A          [[ buffer(0) ]],
                           device const half *B          [[ buffer(1) ]],
                           device half       *C          [[ buffer(2) ]],
                           constant uint     &M_val      [[ buffer(3) ]],
                           constant uint     &N_val      [[ buffer(4) ]],
                           constant uint     &K_val      [[ buffer(5) ]],
                           uint2              gid        [[ threadgroup_position_in_grid ]],
                           uint               lane       [[ thread_index_in_simdgroup ]])
{
    const uint TILE = 8;
    uint tileM = gid.x * TILE;
    uint tileN = gid.y * TILE;

    simdgroup_half8x8 acc;
    simdgroup_half8x8 a_frag;
    simdgroup_half8x8 b_frag;

    acc = simdgroup_half8x8(0);

    for (uint k = 0; k < K_val; k += TILE) {
        simdgroup_load(a_frag, A + tileM * K_val + k, K_val);
        simdgroup_load(b_frag, B + k * N_val + tileN, N_val);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
    }

    simdgroup_store(acc, C + tileM * N_val + tileN, N_val);
}
)";

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters: env vars take priority, then argv, then defaults. */
    int dim = DEFAULT_DIM;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) {
        dim = atoi(env_val);
    } else if (argc >= 2) {
        dim = atoi(argv[1]);
    }
    if (dim <= 0 || (dim % SIMD_TILE) != 0) {
        fprintf(stderr, "Matrix dim must be a positive multiple of %d\n",
                SIMD_TILE);
        return EXIT_FAILURE;
    }

    const char* precision = "fp16";
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) precision = env_val;

    int warmup_iterations = DEFAULT_WARMUP_ITERATIONS;
    env_val = getenv("BENCH_PARAM_warmup_iterations");
    if (env_val) warmup_iterations = atoi(env_val);

    int timed_iterations = DEFAULT_TIMED_ITERATIONS;
    env_val = getenv("BENCH_PARAM_timed_iterations");
    if (env_val) timed_iterations = atoi(env_val);

    int M = dim, N = dim, K = dim;

    @autoreleasepool {
        /* ----------------------------------------------------------------- */
        /* Device & pipeline setup                                           */
        /* ----------------------------------------------------------------- */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this system.\n");
            return EXIT_FAILURE;
        }

        fprintf(stderr, "Device           : %s\n",
               [[device name] UTF8String]);
        fprintf(stderr, "Matrix (M×N×K)   : %d × %d × %d\n", M, N, K);
        fprintf(stderr, "Tile size        : %d × %d (simdgroup)\n", SIMD_TILE, SIMD_TILE);
        fprintf(stderr, "Precision        : %s\n", precision);
        fprintf(stderr, "Warmup iters     : %d\n", warmup_iterations);
        fprintf(stderr, "Timed iters      : %d\n\n", timed_iterations);

        /* Compile shader from embedded source. */
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:metalShaderSource];
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;

        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:opts
                                                        error:&error];
        if (!library) {
            fprintf(stderr, "Shader compile error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return EXIT_FAILURE;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"simdgroup_gemm"];
        if (!function) {
            fprintf(stderr, "Function 'simdgroup_gemm' not found.\n");
            return EXIT_FAILURE;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return EXIT_FAILURE;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        /* ----------------------------------------------------------------- */
        /* Allocate buffers                                                  */
        /* ----------------------------------------------------------------- */
        size_t elemA = (size_t)M * K;
        size_t elemB = (size_t)K * N;
        size_t elemC = (size_t)M * N;

        id<MTLBuffer> bufA = [device newBufferWithLength:elemA * sizeof(__fp16)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:elemB * sizeof(__fp16)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:elemC * sizeof(__fp16)
                                                 options:MTLResourceStorageModeShared];

        /* Initialize with small values (0.01h). */
        __fp16 *pA = (__fp16 *)[bufA contents];
        __fp16 *pB = (__fp16 *)[bufB contents];
        for (size_t i = 0; i < elemA; ++i) pA[i] = (__fp16)0.01f;
        for (size_t i = 0; i < elemB; ++i) pB[i] = (__fp16)0.01f;
        memset([bufC contents], 0, elemC * sizeof(__fp16));

        /* Constant buffers for M, N, K. */
        uint32_t mVal = (uint32_t)M;
        uint32_t nVal = (uint32_t)N;
        uint32_t kVal = (uint32_t)K;

        id<MTLBuffer> bufM = [device newBufferWithBytes:&mVal length:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN = [device newBufferWithBytes:&nVal length:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [device newBufferWithBytes:&kVal length:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];

        /* Grid: one simdgroup per 8×8 tile. */
        MTLSize gridSize = MTLSizeMake(M / SIMD_TILE, N / SIMD_TILE, 1);
        MTLSize tgSize   = MTLSizeMake(SIMDGROUP_SIZE, 1, 1);

        /* ----------------------------------------------------------------- */
        /* Lambda to encode and submit one kernel execution.                 */
        /* ----------------------------------------------------------------- */
        auto run_once = [&]() {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBuffer:bufM offset:0 atIndex:3];
            [enc setBuffer:bufN offset:0 atIndex:4];
            [enc setBuffer:bufK offset:0 atIndex:5];
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        };

        /* ----------------------------------------------------------------- */
        /* Warmup                                                            */
        /* ----------------------------------------------------------------- */
        for (int i = 0; i < warmup_iterations; ++i) {
            run_once();
        }

        /* ----------------------------------------------------------------- */
        /* Timed iterations                                                  */
        /* ----------------------------------------------------------------- */
        double ms_factor = ticks_to_ms();
        uint64_t t0 = mach_absolute_time();

        for (int i = 0; i < timed_iterations; ++i) {
            run_once();
        }

        uint64_t t1 = mach_absolute_time();
        double total_ms = (double)(t1 - t0) * ms_factor;
        double avg_ms   = total_ms / timed_iterations;
        double avg_s    = avg_ms * 1e-3;

        /* TOPS = 2 * M * N * K / time_s / 1e12 */
        double ops  = 2.0 * (double)M * (double)N * (double)K;
        double tops = ops / avg_s / 1e12;

        fprintf(stderr, "fp16     %.3f ms  %.3f TOPS\n", avg_ms, tops);

        /* JSON-lines output to stdout: kernel event */
        printf("{\"type\":\"kernel\",\"name\":\"wmma_fp16_gemm\",\"time_ms\":%.6f,"
               "\"params\":{\"matrix_dim\":%d,\"precision\":\"%s\","
               "\"warmup_iterations\":%d,\"timed_iterations\":%d}}\n",
               avg_ms, dim, precision, warmup_iterations, timed_iterations);

        /* JSON-lines output to stdout: summary */
        printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
               "\"metrics\":[{\"name\":\"TFLOPS\",\"value\":%.3f}]}\n",
               total_ms, tops);

        fprintf(stderr, "\nDone.\n");
    }

    return EXIT_SUCCESS;
}
