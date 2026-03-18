/**
 * polybench_3mm_metal.mm — Apple Metal host for the PolyBench 3mm benchmark.
 *
 * Three chained matrix multiplications (all N×N, single-precision):
 *   E = A * B
 *   F = C * D
 *   G = E * F
 *
 * Reports GFLOPS = total_flops / time_s / 1e9
 *   total_flops = 2*N^3 + 2*N^3 + 2*N^3 = 6*N^3
 *
 * Usage:
 *   ./polybench_3mm [--size N] [--iterations I]
 *
 *   --size N         Matrix dimension N (default: 256)
 *   --iterations I   Timed iterations   (default: 5)
 *
 * Output (stdout): CSV — polybench_3mm,<N>,<GFLOPS>
 * Output (stderr): device info, timing details
 *
 * Build:
 *   make PLATFORM=metal
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double ticks_to_ms(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-6;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static int parseIterations(int argc, char** argv) {
    int iters = 5;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--iterations") == 0) {
            int v = atoi(argv[i + 1]);
            if (v >= 1) iters = v;
            break;
        }
    }
    return iters;
}

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Dispatch helper: run one matrix-multiply kernel
// ---------------------------------------------------------------------------

static void dispatch_mm(id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pso,
                        id<MTLBuffer> left,
                        id<MTLBuffer> right,
                        id<MTLBuffer> out,
                        uint32_t rows, uint32_t inner, uint32_t cols)
{
    uint32_t dims[4] = { rows, inner, cols, 0 };

    id<MTLCommandBuffer>         cb  = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:left  offset:0 atIndex:0];
    [enc setBuffer:right offset:0 atIndex:1];
    [enc setBuffer:out   offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];

    // One thread per output element; Metal clips to grid bounds automatically
    MTLSize threads_per_tg = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads_per_tg];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {

    int N     = parseIntParam(argc, argv, "--size", 256);
    int iters = parseIterations(argc, argv);

    // All matrices are N×N
    const uint32_t NI = (uint32_t)N;
    const uint32_t NJ = (uint32_t)N;
    const uint32_t NK = (uint32_t)N;
    const uint32_t NL = (uint32_t)N;
    const uint32_t NM = (uint32_t)N;

    size_t bytes = (size_t)N * N * sizeof(float);

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Matrix size: %d×%d (all)  |  Iterations: %d\n\n",
            N, N, iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from source file next to the binary
    // -------------------------------------------------------------------
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    NSString *exePath = [[NSBundle mainBundle] executablePath];
    NSString *dir     = [exePath stringByDeletingLastPathComponent];
    NSString *srcPath = [dir stringByAppendingPathComponent:@"polybench_3mm.metal"];

    if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
        NSString *metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        library = [device newLibraryWithSource:metalSrc options:opts error:&err];
        if (!library) {
            fprintf(stderr, "Metal compile error: %s\n",
                    err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Error: polybench_3mm.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    // Load the three kernels
    id<MTLFunction> fn1 = [library newFunctionWithName:@"mm3_kernel1"];
    id<MTLFunction> fn2 = [library newFunctionWithName:@"mm3_kernel2"];
    id<MTLFunction> fn3 = [library newFunctionWithName:@"mm3_kernel3"];
    if (!fn1 || !fn2 || !fn3) {
        fprintf(stderr, "Error: one or more kernels not found in Metal library\n");
        return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> pso1 = [device newComputePipelineStateWithFunction:fn1 error:&err];
    if (!pso1) {
        fprintf(stderr, "Error creating pipeline 1: %s\n", err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso2 = [device newComputePipelineStateWithFunction:fn2 error:&err];
    if (!pso2) {
        fprintf(stderr, "Error creating pipeline 2: %s\n", err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso3 = [device newComputePipelineStateWithFunction:fn3 error:&err];
    if (!pso3) {
        fprintf(stderr, "Error creating pipeline 3: %s\n", err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------
    // Allocate Metal buffers (shared memory)
    // A: NI×NK, B: NK×NJ, C: NJ×NM, D: NM×NL, E: NI×NJ, F: NJ×NL, G: NI×NL
    // Since all dimensions are N, all buffers have the same size.
    // -------------------------------------------------------------------
    id<MTLBuffer> buf_A = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_B = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_C = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_D = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_E = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_F = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_G = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    // Initialize A, B, C, D with random floats
    float* h_A = (float*)buf_A.contents;
    float* h_B = (float*)buf_B.contents;
    float* h_C = (float*)buf_C.contents;
    float* h_D = (float*)buf_D.contents;

    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
        h_C[i] = (float)(rand() % 100) / 10.0f;
        h_D[i] = (float)(rand() % 100) / 10.0f;
    }

    // Lambda to run all three kernels once
    auto run_3mm = [&]() {
        // E = A * B
        dispatch_mm(queue, pso1, buf_A, buf_B, buf_E, NI, NK, NJ);
        // F = C * D
        dispatch_mm(queue, pso2, buf_C, buf_D, buf_F, NJ, NM, NL);
        // G = E * F
        dispatch_mm(queue, pso3, buf_E, buf_F, buf_G, NI, NJ, NL);
    };

    // -------------------------------------------------------------------
    // Warmup (not measured)
    // -------------------------------------------------------------------
    run_3mm();

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = mach_absolute_time();
        run_3mm();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
    }

    // Compute statistics
    double sum_ms = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iters; ++i) {
        sum_ms += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg_ms = sum_ms / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; ++i) {
        double d = times[i] - avg_ms;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;

    // total_flops = 2*NI*NK*NJ + 2*NJ*NM*NL + 2*NI*NJ*NL
    double total_flops = 2.0 * ((double)NI * NK * NJ +
                                (double)NJ * NM * NL +
                                (double)NI * NJ * NL);
    double gflops = total_flops / (avg_ms * 1e-3) / 1e9;

    fprintf(stderr, "Performance: %.2f GFLOPS  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gflops, avg_ms, mn, mx, stddev);

    // CSV output: polybench_3mm,<N>,<GFLOPS>
    printf("polybench_3mm,%d,%.2f\n", N, gflops);

    return EXIT_SUCCESS;

    } // @autoreleasepool
}
