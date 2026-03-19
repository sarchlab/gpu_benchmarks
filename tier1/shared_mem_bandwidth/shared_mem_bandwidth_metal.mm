/**
 * shared_mem_bandwidth_metal.mm
 *
 * Apple Metal port of the shared memory (threadgroup memory) bandwidth
 * microbenchmark.
 *
 * Measures GPU threadgroup memory bandwidth with two access patterns:
 *   - no_conflict : each thread in a SIMD-group accesses a different bank
 *   - conflict    : all threads in a SIMD-group access the same bank
 *
 * Timing is performed using mach_absolute_time for wall-clock accuracy.
 * 5 warmup + 5 timed iterations are performed and the average is reported.
 *
 * Usage:
 *   ./shared_mem_bandwidth
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

static const int BLOCK_SIZE    = 256;           // threads per threadgroup
static const int NUM_GROUPS    = 128;           // threadgroup count
static const int INNER_ITERS   = 1000;          // inner-loop iterations
static const int WARMUP_ITERS  = 5;             // outer warmup dispatches
static const int TIMED_ITERS   = 5;             // outer timed dispatches
static const int SMEM_FLOATS   = 12288;         // 48 KB / 4 bytes
static const double BYTES_PER_GB = 1e9;

// Parameter struct — must match the Metal shader layout
struct Params {
    uint32_t inner_iters;
    uint32_t smem_floats;
    uint32_t threads_per_group;
};

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/** Convert mach_absolute_time ticks to seconds. */
static double ticks_to_seconds(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0)
        mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-9;
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/**
 * Dispatch a named kernel, time it, and return the average time in ms.
 *
 * @param device     MTLDevice to use
 * @param library    Compiled MTLLibrary containing the kernel
 * @param kernelName Kernel function name (e.g. "smem_bw_no_conflict")
 * @param label      Human-readable label for printing
 * @return           Average dispatch time in milliseconds
 */
static double run_benchmark(id<MTLDevice>  device,
                            id<MTLLibrary> library,
                            NSString      *kernelName,
                            const char    *label)
{
    NSError *err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:kernelName];
    if (!fn) {
        fprintf(stderr, "Error: kernel '%s' not found in library\n",
                kernelName.UTF8String);
        exit(EXIT_FAILURE);
    }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline state: %s\n",
                err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Buffers
    id<MTLBuffer> sinkBuf =
        [device newBufferWithLength:NUM_GROUPS * sizeof(float)
                            options:MTLResourceStorageModeShared];

    Params params;
    params.inner_iters     = INNER_ITERS;
    params.smem_floats     = SMEM_FLOATS;
    params.threads_per_group = BLOCK_SIZE;

    NSUInteger smem_bytes = SMEM_FLOATS * sizeof(float);

    NSUInteger tg_size  = BLOCK_SIZE;
    NSUInteger grid_threads = (NSUInteger)NUM_GROUPS * tg_size;

    // Helper block to dispatch one kernel invocation
    void (^dispatch)(void) = ^{
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:sinkBuf offset:0 atIndex:0];
        [enc setBytes:&params length:sizeof(params) atIndex:1];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
        [enc dispatchThreads:MTLSizeMake(grid_threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i)
        dispatch();

    // Timed iterations
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < TIMED_ITERS; ++i)
        dispatch();
    uint64_t t1 = mach_absolute_time();

    double total_s = ticks_to_seconds(t1 - t0);
    double avg_ms  = (total_s / TIMED_ITERS) * 1e3;

    // Bandwidth
    double bytes_per_launch = 2.0 * SMEM_FLOATS * sizeof(float)
                              * (double)INNER_ITERS * NUM_GROUPS;
    double bw_GBs = bytes_per_launch / (avg_ms * 1e-3) / BYTES_PER_GB;

    printf("%-14s : %8.3f ms  %8.2f GB/s\n", label, avg_ms, bw_GBs);
    return avg_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    // Select GPU device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("Device        : %s\n", device.name.UTF8String);
    printf("Block size    : %d threads\n", BLOCK_SIZE);
    printf("Grid size     : %d groups\n", NUM_GROUPS);
    printf("Shared mem    : %d bytes (%d floats)\n",
           SMEM_FLOATS * (int)sizeof(float), SMEM_FLOATS);
    printf("Inner iters   : %d\n", INNER_ITERS);
    printf("Warmup iters  : %d\n", WARMUP_ITERS);
    printf("Timed iters   : %d\n\n", TIMED_ITERS);

    // Compile Metal shader source at runtime
    NSError *err = nil;
    id<MTLLibrary> library = nil;
    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"shared_mem_bandwidth.metal"];
        NSString *metalSrc = nil;

        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:nil];
        }

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
        } else {
            fprintf(stderr,
                    "Error: shared_mem_bandwidth.metal not found next to "
                    "binary at %s\n", srcPath.UTF8String);
            return EXIT_FAILURE;
        }
    }

    // --- No-conflict ---
    double nc_ms = run_benchmark(device, library,
                                 @"smem_bw_no_conflict", "no_conflict");

    // --- Conflict ---
    double c_ms  = run_benchmark(device, library,
                                 @"smem_bw_conflict", "conflict");

    // Bandwidth numbers (recomputed for CSV)
    double bytes_per_launch = 2.0 * SMEM_FLOATS * sizeof(float)
                              * (double)INNER_ITERS * NUM_GROUPS;
    double nc_GBs = bytes_per_launch / (nc_ms * 1e-3) / BYTES_PER_GB;
    double c_GBs  = bytes_per_launch / (c_ms  * 1e-3) / BYTES_PER_GB;

    // CSV output
    printf("shared_mem_bw,no_conflict,%.3f,%.2f\n", nc_ms, nc_GBs);
    printf("shared_mem_bw,conflict,%.3f,%.2f\n",    c_ms,  c_GBs);

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
