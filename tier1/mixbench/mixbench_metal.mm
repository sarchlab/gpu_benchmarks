/**
 * mixbench_metal.mm
 *
 * Apple Metal port of the parametric roofline (mixbench) benchmark.
 *
 * Sweeps the compute-to-memory ratio from memory-bound to compute-bound
 * regimes, tracing out the roofline curve for the GPU.
 *
 * Strategy:
 *   Each Metal thread loads one float (4 bytes) from global memory, performs a
 *   variable-length chain of FP32 FMA operations, and stores the result back
 *   (4 bytes).  Total memory traffic per thread: 8 bytes.
 *
 *   For a given flops_per_byte ratio r:
 *     num_fmas = r * 4   (since each FMA = 2 FLOP, total bytes = 8)
 *     achieved_gflops = (num_elements * num_fmas * 2) / time_s / 1e9
 *
 * Ratios tested: 0.5, 1, 2, 4, 8, 16, 32, 64
 *
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./mixbench [buffer_size_mb]
 *
 *   buffer_size_mb  – optional array size in MiB (default 256)
 *
 * Build:
 *   make PLATFORM=metal
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB  = 256;
static const int    WARMUP_ITERATIONS  = 5;
static const int    TIMED_ITERATIONS   = 5;

/** Flops-per-byte ratios to sweep. */
static const double RATIOS[] = {0.5, 1, 2, 4, 8, 16, 32, 64};
static const int    NUM_RATIOS = sizeof(RATIOS) / sizeof(RATIOS[0]);

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
// Benchmark one ratio
// ---------------------------------------------------------------------------

/**
 * Run the mixbench kernel at a given flops-per-byte ratio.
 *
 * @param device        MTLDevice to use
 * @param pso           Pre-compiled pipeline state for the kernel
 * @param queue         Command queue
 * @param d_data        Device buffer with float data
 * @param num_elements  Number of floats in the buffer
 * @param ratio         Target flops-per-byte ratio
 * @param[out] time_ms  Average kernel time in milliseconds
 * @param[out] gflops   Achieved GFLOPS
 */
static void benchmark_ratio(id<MTLDevice> device,
                            id<MTLComputePipelineState> pso,
                            id<MTLCommandQueue> queue,
                            id<MTLBuffer> d_data,
                            uint32_t num_elements,
                            double ratio,
                            double *time_ms,
                            double *gflops)
{
    /* Compute FMA count for this ratio.
       flops_per_byte = (num_fmas * 2) / 8  =>  num_fmas = ratio * 4 */
    uint32_t num_fmas = (uint32_t)(ratio * 4.0);
    if (num_fmas < 1) num_fmas = 1;

    /* Thread-group configuration. */
    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > 256) tg_size = 256;
    NSUInteger grid_size = (NSUInteger)num_elements;

    /* Helper block: dispatch one kernel invocation. */
    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_data    offset:0 atIndex:0];
        [enc setBytes:&num_elements length:sizeof(num_elements) atIndex:1];
        [enc setBytes:&num_fmas     length:sizeof(num_fmas)     atIndex:2];
        [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Warmup iterations. */
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        dispatch_once();
    }

    /* Timed iterations. */
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < TIMED_ITERATIONS; ++i) {
        dispatch_once();
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;
    double avg_ms = elapsed_ms / TIMED_ITERATIONS;

    /* Total FLOPs across all threads and all timed iterations:
       total_flops = TIMED_ITERATIONS * num_elements * num_fmas * 2 */
    double total_flops = (double)TIMED_ITERATIONS
                         * (double)num_elements
                         * (double)num_fmas
                         * 2.0;
    double achieved_gflops = total_flops / elapsed_s / 1e9;

    *time_ms = avg_ms;
    *gflops  = achieved_gflops;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    /* Parse optional buffer size argument. */
    int buffer_mb = DEFAULT_BUFFER_MB;
    if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
        if (buffer_mb <= 0) {
            fprintf(stderr, "Invalid buffer size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;
    uint32_t num_elements = (uint32_t)(bytes / sizeof(float));

    /* Select the default Metal GPU device. */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("Device          : %s\n", device.name.UTF8String);
    printf("Array size      : %d MiB (%u elements, %zu bytes)\n",
           buffer_mb, num_elements, bytes);
    printf("Warmup iters    : %d\n", WARMUP_ITERATIONS);
    printf("Timed iters     : %d\n", TIMED_ITERATIONS);
    printf("Ratios          : ");
    for (int i = 0; i < NUM_RATIOS; ++i) {
        printf("%.1f%s", RATIOS[i], (i < NUM_RATIOS - 1) ? ", " : "\n");
    }
    printf("\n");

    /* Load the Metal shader library from source (mixbench.metal must be
       in the same directory as the executable). */
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"mixbench.metal"];
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
                    "Error: mixbench.metal not found next to binary.\n"
                    "       Run the benchmark from its build directory.\n");
            return EXIT_FAILURE;
        }
    }

    /* Look up the kernel function and build a pipeline state object. */
    id<MTLFunction> fn = [library newFunctionWithName:@"mixbench_kernel"];
    if (!fn) {
        fprintf(stderr,
                "Error: could not find kernel 'mixbench_kernel' in library\n");
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

    /* Allocate and initialize host data. */
    float *h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory (%zu bytes)\n", bytes);
        return EXIT_FAILURE;
    }
    for (uint32_t i = 0; i < num_elements; ++i) {
        h_data[i] = 1.0f + (float)(i % 1024) * 0.001f;
    }

    /* Allocate shared Metal buffer and copy initial data. */
    id<MTLBuffer> d_data = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!d_data) {
        fprintf(stderr, "Failed to allocate Metal buffer (%zu bytes)\n", bytes);
        free(h_data);
        return EXIT_FAILURE;
    }

    /* Print header. */
    printf("%-16s %10s %12s\n", "FLOP/byte", "Time (ms)", "GFLOPS");
    printf("%-16s %10s %12s\n", "--------", "--------", "--------");

    /* Sweep all ratios. */
    for (int i = 0; i < NUM_RATIOS; ++i) {
        double ratio = RATIOS[i];
        double avg_time_ms = 0.0;
        double achieved_gflops = 0.0;

        /* Re-upload data before each ratio. */
        memcpy(d_data.contents, h_data, bytes);

        benchmark_ratio(device, pso, queue, d_data, num_elements, ratio,
                        &avg_time_ms, &achieved_gflops);

        printf("%-16.1f %10.3f %12.2f\n", ratio, avg_time_ms, achieved_gflops);

        /* CSV output (machine-parseable):
           mixbench,<flops_per_byte>,<time_ms>,<achieved_gflops> */
        printf("mixbench,%.1f,%.3f,%.2f\n", ratio, avg_time_ms, achieved_gflops);
    }

    /* Clean up. */
    free(h_data);

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
