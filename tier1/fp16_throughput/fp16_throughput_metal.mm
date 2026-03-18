/**
 * fp16_throughput_metal.mm
 *
 * Apple Metal port of the fp16_throughput microbenchmark.
 *
 * Measures peak half-precision (FP16) floating-point compute throughput in
 * GFLOPS (10^9 floating-point operations per second) using the Metal Shading
 * Language `half2` type for packed FP16 FMA operations.
 *
 * Strategy:
 *   The Metal compute kernel (fp16_fma in fp16_throughput.metal) executes a
 *   long chain of half2 fma() operations on register-resident values.  The
 *   Metal Shading Language `half2` type holds two fp16 values, so each
 *   fma(half2, half2, half2) call performs two FP16 FMAs simultaneously:
 *
 *       FLOPs per fma(half2, half2, half2) = 2 (packed elements) × 2 = 4
 *
 *   GFLOPS = (num_threads × fmas_per_thread × 4) / elapsed_seconds / 1e9
 *
 * Timing uses mach_absolute_time() for wall-clock accuracy on Apple Silicon.
 *
 * Usage:
 *   ./fp16_throughput [fmas_per_thread]
 *
 *   fmas_per_thread  – optional number of fma() iterations per thread
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

/** Default number of half2 fma() operations per thread per invocation. */
static const int DEFAULT_FMAS_PER_THREAD = 4096;

/** Number of threads per threadgroup. */
static const int THREADS_PER_BLOCK = 256;

/**
 * Number of threadgroups.
 * 1024 × 256 = 262 144 total threads — sufficient to saturate Apple Silicon
 * GPU cores (M1/M2/M3 have 8–38 cores with 1 024 threads each).
 */
static const int NUM_BLOCKS = 1024;

/** How many times to repeat the kernel for a stable average. */
static const int NUM_ITERATIONS = 10;

// ---------------------------------------------------------------------------
// Timing helpers
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

    /* Select GPU device. */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("Device          : %s\n", device.name.UTF8String);
    printf("Grid            : %d blocks × %d threads = %d total threads\n",
           NUM_BLOCKS, THREADS_PER_BLOCK,
           NUM_BLOCKS * THREADS_PER_BLOCK);
    printf("FMAs / thread   : %d  (fma(half2,...), 4 FLOPs each)\n",
           fmas_per_thread);
    printf("Kernel repeats  : %d\n\n", NUM_ITERATIONS);

    /* Compile the Metal shader library at runtime from source file. */
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    /* Look for the .metal source next to the executable. */
    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"fp16_throughput.metal"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            NSString *metalSrc =
                [NSString stringWithContentsOfFile:srcPath
                                          encoding:NSUTF8StringEncoding
                                             error:nil];
            if (metalSrc) {
                MTLCompileOptions *opts = [MTLCompileOptions new];
                library = [device newLibraryWithSource:metalSrc
                                               options:opts
                                                 error:&err];
                if (!library) {
                    fprintf(stderr, "Metal compile error: %s\n",
                            err.localizedDescription.UTF8String);
                    return EXIT_FAILURE;
                }
            }
        }
    }

    if (!library) {
        fprintf(stderr,
                "Error: fp16_throughput.metal not found next to binary.\n"
                "Make sure the .metal file is in the same directory as the "
                "executable.\n");
        return EXIT_FAILURE;
    }

    /* Build pipeline state for the fp16_fma kernel. */
    id<MTLFunction> fn = [library newFunctionWithName:@"fp16_fma"];
    if (!fn) {
        fprintf(stderr, "Error: could not find kernel 'fp16_fma' in library\n");
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline state: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    /* Create command queue. */
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Allocate a tiny output buffer (1 half2 = 4 bytes) for the checksum. */
    id<MTLBuffer> outBuf = [device newBufferWithLength:4
                                              options:MTLResourceStorageModeShared];

    /* Thread-group / grid dimensions. */
    NSUInteger tg_size  = (NSUInteger)THREADS_PER_BLOCK;
    NSUInteger grid_dim = (NSUInteger)NUM_BLOCKS * tg_size;

    /* Warm-up launch — not timed. */
    {
        id<MTLCommandBuffer>       cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:outBuf offset:0 atIndex:0];
        [enc setBytes:&fmas_per_thread length:sizeof(fmas_per_thread) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(grid_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /* Timed benchmark loop. */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        id<MTLCommandBuffer>       cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:outBuf offset:0 atIndex:0];
        [enc setBytes:&fmas_per_thread length:sizeof(fmas_per_thread) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(grid_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    uint64_t t1 = mach_absolute_time();

    /* Compute GFLOPS.
     *   Each fma(half2, half2, half2) performs 2 packed FP16 FMAs.
     *   Each FMA = 2 FLOPs (multiply + add).
     *   => 4 FLOPs per fma(half2,...) call.
     *
     *   total_flops = iterations × num_threads × fmas_per_thread × 4
     */
    long long num_threads = (long long)NUM_BLOCKS * THREADS_PER_BLOCK;
    long long total_flops = (long long)NUM_ITERATIONS
                            * num_threads
                            * (long long)fmas_per_thread
                            * 4LL;
    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;
    double gflops     = (double)total_flops / elapsed_s / 1e9;

    printf("Elapsed time    : %.3f ms (%.3f ms / kernel)\n",
           elapsed_ms, elapsed_ms / NUM_ITERATIONS);
    printf("Total FLOPs     : %lld\n", total_flops);
    printf("FP16 Throughput : %.2f GFLOPS\n", gflops);

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
