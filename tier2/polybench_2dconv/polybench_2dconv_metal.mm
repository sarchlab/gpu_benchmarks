/**
 * polybench_2dconv_metal.mm — Apple Metal host for PolyBench 2D Convolution.
 *
 * Applies a fixed 3×3 kernel to an NI×NJ matrix A, producing output B.
 * PolyBench standard coefficients (embedded in the Metal shader):
 *   c = {{0.8, 0.2, 0.3},
 *        {0.2, 0.7, 0.4},
 *        {0.1, 0.2, 0.5}}
 *
 * Usage:
 *   ./polybench_2dconv [--size N] [--iterations I]
 *
 *   --size N        Matrix dimension N×N        (default: 2048)
 *   --iterations I  Timed benchmark iterations  (default: 5)
 *
 * Output (stdout): CSV — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
 * Output (stderr): GFLOPS
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
// CSV output
// ---------------------------------------------------------------------------

struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

static void printCSVHeader() {
    printf("kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms\n");
}

static void printCSVRow(const BenchResult& r) {
    printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
           r.kernel_name, r.problem_size, r.iterations,
           r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {

    int N     = parseIntParam(argc, argv, "--size", 2048);
    int iters = parseIterations(argc, argv);

    int NI = N, NJ = N;
    size_t bytes = (size_t)NI * NJ * sizeof(float);

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Matrix: %d×%d  |  Iterations: %d\n\n", NI, NJ, iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from source file next to the binary
    // -------------------------------------------------------------------
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    // Look for shader next to the executable
    NSString *exePath = [[NSBundle mainBundle] executablePath];
    NSString *dir     = [exePath stringByDeletingLastPathComponent];
    NSString *srcPath = [dir stringByAppendingPathComponent:@"polybench_2dconv.metal"];

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
        fprintf(stderr, "Error: polybench_2dconv.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"convolution2D_kernel"];
    if (!fn) {
        fprintf(stderr, "Error: kernel 'convolution2D_kernel' not found\n");
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------
    // Allocate buffers (shared memory so host can initialize easily)
    // -------------------------------------------------------------------
    id<MTLBuffer> buf_A = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_B = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];

    // Initialize A with random data
    float* h_A = (float*)buf_A.contents;
    srand(42);
    for (int i = 0; i < NI * NJ; i++)
        h_A[i] = (float)(rand() % 100) / 10.0f;

    // dims: {NJ, NI} — x=columns, y=rows
    uint32_t dims_data[2] = { (uint32_t)NJ, (uint32_t)NI };

    // Thread dispatch: one thread per element, 16×16 threadgroup
    NSUInteger tg_x = 16, tg_y = 16;
    NSUInteger grid_x = (NSUInteger)((NJ + 15) / 16) * 16;
    NSUInteger grid_y = (NSUInteger)((NI + 15) / 16) * 16;

    // Helper lambda to dispatch one kernel
    auto dispatch_kernel = [&]() {
        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_A offset:0 atIndex:0];
        [enc setBuffer:buf_B offset:0 atIndex:1];
        [enc setBytes:dims_data length:sizeof(dims_data) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(grid_x, grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // Warmup (not measured)
    dispatch_kernel();

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int iter = 0; iter < iters; ++iter) {
        uint64_t t0 = mach_absolute_time();
        dispatch_kernel();
        uint64_t t1 = mach_absolute_time();
        times[iter] = ticks_to_ms(t1 - t0);
    }

    double sum = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iters; ++i) {
        sum += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;

    char problemSize[64];
    snprintf(problemSize, sizeof(problemSize), "%d", N);

    BenchResult r;
    r.kernel_name  = "polybench_2dconv";
    r.problem_size = problemSize;
    r.iterations   = iters;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;

    printCSVHeader();
    printCSVRow(r);

    // GFLOPS: each interior output element = 9 MACs = 18 FLOPs
    long long interior = (long long)(NI - 2) * (NJ - 2);
    double gflops = 2.0 * 9.0 * (double)interior / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "GFLOPS: %.2f  (avg %.4f ms)\n", gflops, r.avg_ms);

    return EXIT_SUCCESS;

    } // @autoreleasepool
}
