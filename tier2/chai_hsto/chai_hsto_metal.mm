/**
 * chai_hsto_metal.mm — Apple Metal host for histogram benchmark.
 *
 * Computes a 256-bin histogram of N random bytes using threadgroup memory
 * privatization and atomic reduction.
 *
 * Usage:
 *   ./chai_hsto [--size N] [--iterations I]
 *
 *   --size N         Number of input bytes (default: 16777216 = 16M)
 *   --iterations I   Timed iterations (default: 5)
 *
 * Output (stdout): CSV — chai_hsto,<N>,<time_ms>,<GBs>
 * Output (stderr): human-readable results
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
#include <vector>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double ticks_to_ms(uint64_t ticks) {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-6;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

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
// Constants
// ---------------------------------------------------------------------------

#define NUM_BINS 256
#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// CPU reference: histogram
// ---------------------------------------------------------------------------

static void histogram_cpu(const unsigned char* data, unsigned int* histo, int N) {
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < N; ++i) {
        histo[data[i]]++;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    @autoreleasepool {

    int N     = parseIntParam(argc, argv, "--size", 16777216);  // 16M
    int iters = parseIntParam(argc, argv, "--iterations", 5);

    size_t data_bytes  = (size_t)N * sizeof(unsigned char);
    size_t histo_bytes = NUM_BINS * sizeof(unsigned int);

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Histogram  |  N: %d (%.2f MB)  |  Bins: %d  |  "
            "Iterations: 5 warmup + %d timed\n\n",
            N, (double)data_bytes / (1024.0 * 1024.0), NUM_BINS, iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader
    // -------------------------------------------------------------------
    NSError* err = nil;
    id<MTLLibrary> library = nil;

    NSString* exePath = [[NSBundle mainBundle] executablePath];
    NSString* dir     = [exePath stringByDeletingLastPathComponent];
    NSString* srcPath = [dir stringByAppendingPathComponent:@"chai_hsto.metal"];

    if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
        NSString* metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        library = [device newLibraryWithSource:metalSrc options:opts error:&err];
        if (!library) {
            fprintf(stderr, "Metal compile error: %s\n",
                    err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Error: chai_hsto.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fnKernel = [library newFunctionWithName:@"histogram_kernel"];
    if (!fnKernel) {
        fprintf(stderr, "Error: kernel function not found\n");
        return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fnKernel error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------
    // Generate random input data (deterministic seed)
    // -------------------------------------------------------------------
    std::vector<unsigned char> h_data(N);
    unsigned int seed = 12345;
    for (int i = 0; i < N; ++i) {
        seed = seed * 1103515245u + 12345u;
        h_data[i] = (unsigned char)((seed >> 16) & 0xFF);
    }

    // -------------------------------------------------------------------
    // Allocate Metal buffers
    // -------------------------------------------------------------------
    id<MTLBuffer> bufData = [device newBufferWithBytes:h_data.data()
                                               length:data_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufHisto = [device newBufferWithLength:histo_bytes
                                                options:MTLResourceStorageModeShared];

    uint32_t paramN = (uint32_t)N;

    NSUInteger tgSize = BLOCK_SIZE;
    // Grid size: enough threads with grid-stride, cap to reasonable number
    NSUInteger gridN = (NSUInteger)((N + tgSize - 1) / tgSize);
    if (gridN > 1024) gridN = 1024;
    gridN *= tgSize;

    // Lambda: run histogram
    auto run_histo = [&]() {
        // Clear histogram
        memset(bufHisto.contents, 0, histo_bytes);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufData  offset:0 atIndex:0];
        [enc setBuffer:bufHisto offset:0 atIndex:1];
        [enc setBytes:&paramN length:sizeof(paramN) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(gridN, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // -------------------------------------------------------------------
    // Warmup (5 iterations)
    // -------------------------------------------------------------------
    for (int w = 0; w < 5; ++w) {
        run_histo();
    }

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = mach_absolute_time();
        run_histo();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
    }

    double sum = 0.0;
    for (int i = 0; i < iters; ++i) sum += times[i];
    double avg_ms = sum / iters;

    // GB/s = N * sizeof(uint8_t) / time_s / 1e9
    double gbs = (double)N * 1.0 / (avg_ms * 1e-3) / 1e9;

    // CSV output to stdout
    printf("chai_hsto,%d,%.4f,%.4f\n", N, avg_ms, gbs);

    // Human-readable output to stderr
    fprintf(stderr, "Average time: %.4f ms\n", avg_ms);
    fprintf(stderr, "Throughput:   %.4f GB/s\n", gbs);

    // -------------------------------------------------------------------
    // Verification (compare GPU histogram against CPU reference)
    // -------------------------------------------------------------------
    {
        const unsigned int* gpuHisto = (const unsigned int*)bufHisto.contents;

        unsigned int cpu_histo[NUM_BINS];
        histogram_cpu(h_data.data(), cpu_histo, N);

        int errors = 0;
        for (int b = 0; b < NUM_BINS; ++b) {
            if (gpuHisto[b] != cpu_histo[b]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at bin %d: GPU=%u CPU=%u\n",
                            b, gpuHisto[b], cpu_histo[b]);
                }
                errors++;
            }
        }

        if (errors > 0)
            fprintf(stderr, "FAIL: %d bin errors out of %d bins\n",
                    errors, NUM_BINS);
        else
            fprintf(stderr, "PASS\n");
    }

    return EXIT_SUCCESS;

    } // @autoreleasepool
}
