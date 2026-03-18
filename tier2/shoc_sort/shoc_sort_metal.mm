/**
 * shoc_sort_metal.mm — Apple Metal host for the SHOC Sort benchmark.
 *
 * Bitonic sort on a large array of random unsigned integers.
 * N = 1<<22 (4M elements) by default.
 *
 * Algorithm: bitonic sort — O(n log^2 n) compare-and-swap network.
 * Each dispatch pass performs one compare-and-swap step.
 *
 * Usage:
 *   ./shoc_sort [--size N] [--iterations I]
 *
 *   --size N         Number of elements (power of 2, default: 4194304)
 *   --iterations I   Timed iterations (default: 5)
 *
 * Output (stdout): CSV — shoc_sort,<N>,<Melements/s>
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

static double ticks_to_seconds(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-9;
}

static double ticks_to_ms(uint64_t ticks)
{
    return ticks_to_seconds(ticks) * 1e3;
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
// Verify: check array is sorted
// ---------------------------------------------------------------------------

static bool verify_sorted(const uint32_t* arr, int N) {
    for (int i = 1; i < N; ++i) {
        if (arr[i] < arr[i - 1]) {
            fprintf(stderr, "Mismatch at %d: arr[%d]=%u > arr[%d]=%u\n",
                    i, i - 1, arr[i - 1], i, arr[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {

    int N     = parseIntParam(argc, argv, "--size", 1 << 22); // 4M default
    int iters = parseIterations(argc, argv);

    // Round N up to next power of 2
    {
        int p = 1;
        while (p < N) p <<= 1;
        if (p != N) {
            fprintf(stderr, "Warning: N=%d is not a power of 2, rounding up to %d\n", N, p);
            N = p;
        }
    }

    size_t bytes = (size_t)N * sizeof(uint32_t);

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Array size: %d elements (%.1f MB) | Iterations: %d\n\n",
            N, (double)bytes / (1024.0 * 1024.0), iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from source file next to the binary
    // -------------------------------------------------------------------
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    NSString *exePath = [[NSBundle mainBundle] executablePath];
    NSString *dir     = [exePath stringByDeletingLastPathComponent];
    NSString *srcPath = [dir stringByAppendingPathComponent:@"shoc_sort.metal"];

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
        fprintf(stderr, "Error: shoc_sort.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"bitonic_step"];
    if (!fn) {
        fprintf(stderr, "Error: kernel 'bitonic_step' not found\n");
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
    // Allocate Metal buffers (shared memory)
    // -------------------------------------------------------------------
    id<MTLBuffer> buf_data   = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_params = [device newBufferWithLength:2 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];

    // Host pointer for initialization and readback
    uint32_t* h_data   = (uint32_t*)buf_data.contents;
    uint32_t* h_params = (uint32_t*)buf_params.contents;

    // Original data for re-initialization each iteration
    uint32_t* h_orig = (uint32_t*)malloc(bytes);
    if (!h_orig) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_orig[i] = (uint32_t)rand();
    }

    // Max threadgroup size for the kernel
    NSUInteger maxTG = pso.maxTotalThreadsPerThreadgroup;
    if (maxTG > 1024) maxTG = 1024;
    NSUInteger tg_size = maxTG;

    // Helper: dispatch one bitonic_step(j, k) pass
    auto dispatch_step = [&](uint32_t j, uint32_t k) {
        h_params[0] = j;
        h_params[1] = k;

        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_data   offset:0 atIndex:0];
        [enc setBuffer:buf_params offset:0 atIndex:1];
        // Each thread handles one element
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // Helper: run one full bitonic sort
    auto run_sort = [&]() {
        for (int k = 2; k <= N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                dispatch_step((uint32_t)j, (uint32_t)k);
            }
        }
    };

    // -------------------------------------------------------------------
    // Warmup (not measured)
    // -------------------------------------------------------------------
    memcpy(h_data, h_orig, bytes);
    run_sort();

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        // Fresh input each iteration
        memcpy(h_data, h_orig, bytes);

        uint64_t t0 = mach_absolute_time();
        run_sort();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
        fprintf(stderr, "  Iter %d: %.4f ms\n", i + 1, times[i]);
    }

    // Statistics
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

    double melems_per_sec = (double)N / (avg_ms * 1e-3) / 1e6;

    fprintf(stderr, "Throughput: %.2f Melements/s  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            melems_per_sec, avg_ms, mn, mx, stddev);

    // Correctness check (last iteration result still in buf_data)
    if (verify_sorted(h_data, N)) {
        fprintf(stderr, "Correctness: PASS\n");
    } else {
        fprintf(stderr, "Correctness: FAIL\n");
    }

    // CSV output: shoc_sort,<N>,<Melements/s>
    printf("shoc_sort,%d,%.2f\n", N, melems_per_sec);

    free(h_orig);
    return EXIT_SUCCESS;

    } // @autoreleasepool
}
