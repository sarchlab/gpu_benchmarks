/**
 * warp_shuffle_metal.mm
 *
 * Apple Metal port of the warp_shuffle microbenchmark.
 *
 * Microbenchmark for SIMD shuffle throughput on Apple GPUs:
 *   - simd_shuffle_xor   (butterfly / XOR shuffle)
 *   - simd_shuffle        (indexed shuffle)
 *   - simd_shuffle_up     (shift up)
 *   - simd_shuffle_down   (shift down)
 *
 * Each thread performs N shuffle operations in a tight loop, measuring
 * the effective data movement rate:
 *
 *     GB/s = (num_threads × ops_per_thread × sizeof(int))
 *            / elapsed_seconds / 1e9
 *
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./warp_shuffle [ops_per_thread]
 *
 *   ops_per_thread  – optional number of shuffle iterations per thread
 *                     (default 4096).
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

/** Default number of shuffle operations per thread per kernel invocation. */
static const int DEFAULT_OPS_PER_THREAD = 4096;

/** Thread-group / grid configuration (matches HIP version). */
static const int NUM_BLOCKS        = 1024;
static const int THREADS_PER_BLOCK = 256;
static const int TOTAL_THREADS     = NUM_BLOCKS * THREADS_PER_BLOCK;

/** Metal SIMD width on Apple GPUs. */
static const int SIMD_WIDTH = 32;

/** Number of warm-up iterations. */
static const int NUM_WARMUP = 5;

/** Number of timed iterations. */
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
// Benchmark helper
// ---------------------------------------------------------------------------

/**
 * Run a single shuffle benchmark: warmup + timed iterations.
 *
 * @param name          Human-readable name of the variant.
 * @param csv_variant   Short name for CSV output.
 * @param fn_name       Name of the Metal kernel function.
 * @param device        Metal device.
 * @param library       Compiled Metal library.
 * @param queue         Command queue.
 * @param d_out         Device buffer for the output int (prevents DCE).
 * @param ops_per_thread Number of shuffle ops per thread.
 */
static int run_benchmark(const char *name, const char *csv_variant,
                         const char *fn_name,
                         id<MTLDevice> device,
                         id<MTLLibrary> library,
                         id<MTLCommandQueue> queue,
                         id<MTLBuffer> d_out,
                         int ops_per_thread)
{
    NSError *err = nil;

    /* Look up the kernel function and build a pipeline. */
    NSString *fnStr = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [library newFunctionWithName:fnStr];
    if (!fn) {
        fprintf(stderr, "Error: could not find kernel '%s' in library\n",
                fn_name);
        return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline state for '%s': %s\n",
                fn_name, err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    /* Thread-group configuration. */
    NSUInteger tg_size = (NSUInteger)THREADS_PER_BLOCK;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;

    uint32_t opt_u32 = (uint32_t)ops_per_thread;

    /* Helper lambda: dispatch one kernel invocation. */
    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_out offset:0 atIndex:0];
        [enc setBytes:&opt_u32 length:sizeof(opt_u32) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)TOTAL_THREADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Warm-up launches. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        dispatch_once();
    }

    /* Timed benchmark loop. */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dispatch_once();
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    /* Compute throughput.
     *   data_moved = iterations * num_threads * ops_per_thread * sizeof(int)
     *   GB/s       = data_moved / elapsed_s / 1e9
     */
    long long num_threads = (long long)TOTAL_THREADS;
    long long total_ops   = (long long)NUM_ITERATIONS
                            * num_threads
                            * (long long)ops_per_thread;
    double data_bytes     = (double)total_ops * sizeof(int);
    double gbs            = data_bytes / elapsed_s / 1e9;

    double avg_ms = elapsed_ms / NUM_ITERATIONS;

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f GB/s\n",
            name, elapsed_ms, avg_ms, gbs);

    /* CSV output to stdout: warp_shuffle,<variant>,<time_ms>,<GBs> */
    printf("warp_shuffle,%s,%.3f,%.2f\n", csv_variant, avg_ms, gbs);

    return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    /* Parse optional argument. */
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
        if (ops_per_thread <= 0) {
            fprintf(stderr, "Invalid ops_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
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
    fprintf(stderr, "SIMD width      : %d\n", SIMD_WIDTH);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n", NUM_WARMUP);
    fprintf(stderr, "Timed iters     : %d\n\n", NUM_ITERATIONS);

    /* Load the Metal shader library from source. */
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"warp_shuffle.metal"];
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
                    "Error: warp_shuffle.metal not found next to binary.\n"
                    "       Run the benchmark from its build directory.\n");
            return EXIT_FAILURE;
        }
    }

    /* Command queue. */
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Allocate a single int buffer for the output (prevents DCE). */
    id<MTLBuffer> d_out =
        [device newBufferWithLength:sizeof(int)
                            options:MTLResourceStorageModeShared];

    /* Run each shuffle variant benchmark. */
    int rc;
    rc = run_benchmark("shfl_xor", "shfl_xor", "shfl_xor_kern",
                       device, library, queue, d_out, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    rc = run_benchmark("shfl", "shfl", "shfl_kern",
                       device, library, queue, d_out, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    rc = run_benchmark("shfl_up", "shfl_up", "shfl_up_kern",
                       device, library, queue, d_out, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    rc = run_benchmark("shfl_down", "shfl_down", "shfl_down_kern",
                       device, library, queue, d_out, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
