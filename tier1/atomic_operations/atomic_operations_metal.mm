/**
 * atomic_operations_metal.mm
 *
 * Apple Metal port of the atomic_operations microbenchmark.
 *
 * Microbenchmark for atomic operation throughput on global memory:
 *   - atomicAdd  (atomic_fetch_add_explicit)
 *   - atomicCAS  (atomic_compare_exchange_weak_explicit)
 *   - atomicMin  (atomic_fetch_min_explicit)
 *
 * All threads perform N atomic operations on a single device-memory
 * atomic_uint, creating maximum contention to measure hardware atomic
 * unit throughput.
 *
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./atomic_operations [ops_per_thread]
 *
 *   ops_per_thread  – optional number of atomic ops per thread
 *                     (default 1024).
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

/** Default number of atomic operations per thread per kernel invocation. */
static const int DEFAULT_OPS_PER_THREAD = 1024;

/** Thread-group / grid configuration (matches HIP version). */
static const int NUM_BLOCKS        = 1024;
static const int THREADS_PER_BLOCK = 256;
static const int TOTAL_THREADS     = NUM_BLOCKS * THREADS_PER_BLOCK;

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
 * Run a single atomic benchmark: warmup + timed iterations.
 *
 * @param name       Human-readable name of the operation.
 * @param csv_op     Short name for CSV output (e.g. "atomicAdd").
 * @param fn_name    Name of the Metal kernel function.
 * @param device     Metal device.
 * @param library    Compiled Metal library.
 * @param queue      Command queue.
 * @param d_counter  Device buffer for the shared atomic counter.
 * @param ops_per_thread Number of atomic ops per thread.
 */
static int run_benchmark(const char *name, const char *csv_op,
                         const char *fn_name,
                         id<MTLDevice> device,
                         id<MTLLibrary> library,
                         id<MTLCommandQueue> queue,
                         id<MTLBuffer> d_counter,
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
        [enc setBuffer:d_counter offset:0 atIndex:0];
        [enc setBytes:&opt_u32 length:sizeof(opt_u32) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)TOTAL_THREADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Reset counter. */
    memset(d_counter.contents, 0, sizeof(uint32_t));

    /* Warm-up launches. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        dispatch_once();
    }

    /* Reset counter before timed run. */
    memset(d_counter.contents, 0, sizeof(uint32_t));

    /* Timed benchmark loop. */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dispatch_once();
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    /* Compute throughput.
     *   total_ops = iterations * num_threads * ops_per_thread
     */
    long long num_threads = (long long)TOTAL_THREADS;
    long long total_ops   = (long long)NUM_ITERATIONS
                            * num_threads
                            * (long long)ops_per_thread;
    double gops = (double)total_ops / elapsed_s / 1e9;

    double avg_ms = elapsed_ms / NUM_ITERATIONS;

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f Gops/s\n",
            name, elapsed_ms, avg_ms, gops);

    /* CSV output to stdout: atomic_ops,<op>_global,<time_ms>,<Gops> */
    printf("atomic_ops,%s_global,%.3f,%.2f\n", csv_op, avg_ms, gops);

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
                             @"atomic_operations.metal"];
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
                    "Error: atomic_operations.metal not found next to binary.\n"
                    "       Run the benchmark from its build directory.\n");
            return EXIT_FAILURE;
        }
    }

    /* Command queue. */
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Allocate a single uint32 buffer for the shared atomic counter. */
    id<MTLBuffer> d_counter =
        [device newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];

    /* Run each atomic operation benchmark. */
    int rc;
    rc = run_benchmark("atomicAdd", "atomicAdd", "atomicAdd_kern",
                       device, library, queue, d_counter, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    rc = run_benchmark("atomicCAS", "atomicCAS", "atomicCAS_kern",
                       device, library, queue, d_counter, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    rc = run_benchmark("atomicMin", "atomicMin", "atomicMin_kern",
                       device, library, queue, d_counter, ops_per_thread);
    if (rc != EXIT_SUCCESS) return rc;

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
