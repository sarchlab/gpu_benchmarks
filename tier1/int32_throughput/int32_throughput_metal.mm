/**
 * int32_throughput_metal.mm
 *
 * Apple Metal port of the int32_throughput microbenchmark.
 *
 * Microbenchmark for peak 32-bit integer (INT32) compute throughput on
 * Apple GPUs (M-series and later).
 *
 * Strategy:
 *   Each Metal thread executes a long chain of integer multiply-add (MAD)
 *   instructions: a = a * mul + add.  Because each MAD counts as
 *   2 integer operations (1 multiply + 1 add), the achieved GIOPS is:
 *
 *       GIOPS = (num_threads × mads_per_thread × 2) / elapsed_seconds / 1e9
 *
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure INT32 arithmetic throughput
 *   rather than memory bandwidth.
 *
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./int32_throughput [mads_per_thread]
 *
 *   mads_per_thread  – optional number of MAD iterations per thread
 *                      (default 4096).  Increase if the kernel finishes in
 *                      under ~5 ms on a fast GPU.
 *
 * Build:
 *   make PLATFORM=metal
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default number of MAD operations per thread per kernel invocation. */
static const int DEFAULT_MADS_PER_THREAD = 4096;

/** Number of threads to dispatch (matches the CUDA/HIP version). */
static const int NUM_BLOCKS        = 1024;
static const int THREADS_PER_BLOCK = 256;
static const int TOTAL_THREADS     = NUM_BLOCKS * THREADS_PER_BLOCK;

/** How many warmup iterations before timing. */
static const int NUM_WARMUP = 5;

/** How many times to repeat the kernel for a stable average. */
static const int NUM_ITERATIONS = 5;

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

/** Convert mach_absolute_time ticks to seconds. */
static double ticks_to_seconds(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-9;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    /* Parse optional argument. */
    int mads_per_thread = DEFAULT_MADS_PER_THREAD;
    if (argc >= 2) {
        mads_per_thread = atoi(argv[1]);
        if (mads_per_thread <= 0) {
            fprintf(stderr, "Invalid mads_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        /* Round down to nearest multiple of 4 to match the loop unroll. */
        mads_per_thread = (mads_per_thread / 4) * 4;
        if (mads_per_thread == 0) mads_per_thread = 4;
    }

    /* Select the default Metal GPU device. */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device          : %s\n", device.name.UTF8String);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);
    fprintf(stderr, "MADs / thread   : %d\n", mads_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n", NUM_WARMUP);
    fprintf(stderr, "Timed iters     : %d\n\n", NUM_ITERATIONS);

    /* Load the Metal shader library from source (int32_throughput.metal must
       be in the same directory as the executable). */
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"int32_throughput.metal"];
        NSString *metalSrc = nil;
        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:nil];
        }

        if (metalSrc) {
            MTLCompileOptions *opts = [MTLCompileOptions new];
            library = [device newLibraryWithSource:metalSrc options:opts error:&err];
            if (!library) {
                fprintf(stderr, "Metal compile error: %s\n",
                        err.localizedDescription.UTF8String);
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr,
                    "Error: int32_throughput.metal not found next to binary.\n"
                    "       Run the benchmark from its build directory.\n");
            return EXIT_FAILURE;
        }
    }

    /* Look up the kernel function and build a pipeline state object. */
    id<MTLFunction> fn = [library newFunctionWithName:@"int32_mad"];
    if (!fn) {
        fprintf(stderr, "Error: could not find kernel 'int32_mad' in library\n");
        return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline state: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    /* Command queue. */
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Allocate a one-int output buffer for the checksum (prevents DCE). */
    id<MTLBuffer> d_out = [device newBufferWithLength:sizeof(int)
                                             options:MTLResourceStorageModeShared];

    /* Thread-group configuration.
     * Use THREADS_PER_BLOCK as the threadgroup size (capped to hardware max),
     * and dispatch exactly TOTAL_THREADS threads in the grid. */
    NSUInteger tg_size = (NSUInteger)THREADS_PER_BLOCK;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;

    /* mads_per_thread is passed as a Metal buffer constant. */
    uint32_t mpt_u32 = (uint32_t)mads_per_thread;

    /* Helper lambda: dispatch one kernel invocation. */
    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_out    offset:0 atIndex:0];
        [enc setBytes:&mpt_u32  length:sizeof(mpt_u32) atIndex:1];
        /* Dispatch a 1-D grid of TOTAL_THREADS threads. */
        [enc dispatchThreads:MTLSizeMake((NSUInteger)TOTAL_THREADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Warm-up launches — not timed. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        dispatch_once();
    }

    /* Timed benchmark loop. */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dispatch_once();
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    /* Compute GIOPS.
     *   total_ops = iterations × num_threads × mads_per_thread × 2
     *   (factor of 2: each MAD is 1 multiply + 1 add)
     */
    long long num_threads = (long long)TOTAL_THREADS;
    long long total_ops   = (long long)NUM_ITERATIONS
                            * num_threads
                            * (long long)mads_per_thread
                            * 2LL;
    double giops = (double)total_ops / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms (%.3f ms / kernel)\n",
           elapsed_ms, elapsed_ms / NUM_ITERATIONS);
    fprintf(stderr, "Total INT ops   : %lld\n", total_ops);
    fprintf(stderr, "INT32 Throughput : %.2f GIOPS\n", giops);

    // CSV output (machine-parseable): benchmark_name,n_ops,time_ms,giops
    printf("int32_throughput,%lld,%.3f,%.2f\n", total_ops, elapsed_ms, giops);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
