/**
 * fp64_throughput_metal.mm
 *
 * Apple Metal port of the fp64_throughput microbenchmark.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  IMPORTANT: Apple GPUs do NOT have native FP64 (double-precision)       │
 * │  hardware.  Metal does not expose a `double` type on Apple Silicon.     │
 * │  This benchmark therefore runs FP32 FMA operations as a proxy and       │
 * │  reports "FP32 Throughput (FP64 proxy)" to make the limitation clear.  │
 * │                                                                         │
 * │  True FP64 throughput measurements require NVIDIA or AMD GPUs.          │
 * │  Use the CUDA or HIP build (PLATFORM=cuda or PLATFORM=rocm) for those. │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Strategy:
 *   Each Metal thread executes a long chain of fused multiply-add (FMA)
 *   instructions via the Metal Shading Language fma() builtin on float values.
 *   Because each FMA counts as 2 floating-point operations (1 multiply + 1 add),
 *   the achieved GFLOPS is:
 *
 *       GFLOPS = (num_threads × fmas_per_thread × 2) / elapsed_seconds / 1e9
 *
 *   The kernel is intentionally free of memory traffic: all values live in
 *   registers, so the measurement reflects pure FP32 arithmetic throughput.
 *
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./fp64_throughput [fmas_per_thread]
 *
 *   fmas_per_thread  – optional number of FMA iterations per thread
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

/** Default number of FMA operations per thread per kernel invocation. */
static const int DEFAULT_FMAS_PER_THREAD = 4096;

/** Number of threads to dispatch (matches the CUDA/HIP version). */
static const int NUM_BLOCKS        = 1024;
static const int THREADS_PER_BLOCK = 256;
static const int TOTAL_THREADS     = NUM_BLOCKS * THREADS_PER_BLOCK;

/** How many times to repeat the kernel for a stable average. */
static const int NUM_ITERATIONS = 10;

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
    int fmas_per_thread = DEFAULT_FMAS_PER_THREAD;
    if (argc >= 2) {
        fmas_per_thread = atoi(argv[1]);
        if (fmas_per_thread <= 0) {
            fprintf(stderr, "Invalid fmas_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        /* Round down to nearest multiple of 4 to match the loop unroll. */
        fmas_per_thread = (fmas_per_thread / 4) * 4;
        if (fmas_per_thread == 0) fmas_per_thread = 4;
    }

    /* Select the default Metal GPU device. */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("Device          : %s\n", device.name.UTF8String);
    printf("NOTE            : Apple GPUs lack native FP64 hardware.\n");
    printf("                  Measuring FP32 throughput as proxy.\n\n");
    printf("Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);
    printf("FMAs / thread   : %d\n", fmas_per_thread);
    printf("Kernel repeats  : %d\n\n", NUM_ITERATIONS);

    /* Load the Metal shader library from source (fp64_throughput.metal must
       be in the same directory as the executable). */
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"fp64_throughput.metal"];
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
                    "Error: fp64_throughput.metal not found next to binary.\n"
                    "       Run the benchmark from its build directory.\n");
            return EXIT_FAILURE;
        }
    }

    /* Look up the kernel function and build a pipeline state object. */
    id<MTLFunction> fn = [library newFunctionWithName:@"fp64_fma"];
    if (!fn) {
        fprintf(stderr, "Error: could not find kernel 'fp64_fma' in library\n");
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

    /* Allocate a one-float output buffer for the checksum (prevents DCE). */
    id<MTLBuffer> d_out = [device newBufferWithLength:sizeof(float)
                                             options:MTLResourceStorageModeShared];

    /* Thread-group configuration.
     * Use THREADS_PER_BLOCK as the threadgroup size (capped to hardware max),
     * and dispatch exactly TOTAL_THREADS threads in the grid. */
    NSUInteger tg_size = (NSUInteger)THREADS_PER_BLOCK;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;

    /* fmas_per_thread is passed as a Metal buffer constant. */
    uint32_t fpt_u32 = (uint32_t)fmas_per_thread;

    /* Helper lambda: dispatch one kernel invocation. */
    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_out    offset:0 atIndex:0];
        [enc setBytes:&fpt_u32  length:sizeof(fpt_u32) atIndex:1];
        /* Dispatch a 1-D grid of TOTAL_THREADS threads. */
        [enc dispatchThreads:MTLSizeMake((NSUInteger)TOTAL_THREADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Warm-up launch — not timed. */
    dispatch_once();

    /* Timed benchmark loop. */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dispatch_once();
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    /* Compute GFLOPS.
     *   total_flops = iterations × num_threads × fmas_per_thread × 2
     *   (factor of 2: each FMA is 1 multiply + 1 add)
     */
    long long num_threads = (long long)TOTAL_THREADS;
    long long total_flops = (long long)NUM_ITERATIONS
                            * num_threads
                            * (long long)fmas_per_thread
                            * 2LL;
    double gflops = (double)total_flops / elapsed_s / 1e9;

    printf("Elapsed time    : %.3f ms (%.3f ms / kernel)\n",
           elapsed_ms, elapsed_ms / NUM_ITERATIONS);
    printf("Total FLOPs     : %lld\n", total_flops);
    /* Label clearly as FP32 proxy since Apple GPUs have no native FP64. */
    printf("FP32 Throughput : %.2f GFLOPS  (FP64 proxy — Apple GPUs lack native FP64)\n",
           gflops);

    // CSV output (machine-parseable): benchmark_name,n_ops,time_ms,gflops
    printf("fp64_throughput,%lld,%.3f,%.2f\n", total_flops, elapsed_ms, gflops);

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
