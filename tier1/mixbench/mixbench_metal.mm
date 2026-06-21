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
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./mixbench [buffer_size_mb]
 *
 *   buffer_size_mb  – optional array size in MiB (default 256)
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_flops_per_byte
 *   BENCH_PARAM_threads_per_block
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
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB       = 256;
static const double DEFAULT_FLOPS_PER_BYTE  = 4.0;
static const int    DEFAULT_THREADS_PER_BLK = 256;
static const int    DEFAULT_WARMUP_ITERS     = 5;

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
 */
static void benchmark_ratio(id<MTLDevice> device,
                            id<MTLComputePipelineState> pso,
                            id<MTLCommandQueue> queue,
                            id<MTLBuffer> d_data,
                            uint32_t num_elements,
                            double ratio,
                            int threads_per_block,
                            int warmup_iters,
                            double *time_ms,
                            double *gflops,
                            double *bw_gbps)
{
    /* Compute FMA count for this ratio. */
    uint32_t num_fmas = (uint32_t)(ratio * 4.0);
    if (num_fmas < 1) num_fmas = 1;

    /* Thread-group configuration. */
    NSUInteger tg_size = (NSUInteger)threads_per_block;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;
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
    for (int i = 0; i < warmup_iters; ++i) {
        dispatch_once();
    }

    /* Timed kernel launch. */
    uint64_t t0 = mach_absolute_time();
    dispatch_once();
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;
    double avg_ms = elapsed_ms;

    /* Total FLOPs */
    double total_flops = (double)num_elements
                         * (double)num_fmas
                         * 2.0;
    double achieved_gflops = total_flops / elapsed_s / 1e9;

    /* Bandwidth: each element = 8 bytes (4 read + 4 write) */
    double total_bytes = (double)num_elements * 8.0;
    double achieved_bw = total_bytes / elapsed_s / 1e9;

    *time_ms = avg_ms;
    *gflops  = achieved_gflops;
    *bw_gbps = achieved_bw;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters: env vars take priority, then argv, then defaults. */
    int buffer_mb = DEFAULT_BUFFER_MB;
    env_val = getenv("BENCH_PARAM_buffer_size_mb");
    if (env_val) {
        buffer_mb = atoi(env_val);
    } else if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
    }
    if (buffer_mb <= 0) {
        fprintf(stderr, "Invalid buffer_size_mb\n");
        return EXIT_FAILURE;
    }

    double flops_per_byte = DEFAULT_FLOPS_PER_BYTE;
    env_val = getenv("BENCH_PARAM_flops_per_byte");
    if (env_val) flops_per_byte = atof(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);


    int num_warmup = DEFAULT_WARMUP_ITERS;

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;
    uint32_t num_elements = (uint32_t)(bytes / sizeof(float));

    /* Select the default Metal GPU device. */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device          : %s\n", device.name.UTF8String);
    fprintf(stderr, "Array size      : %d MiB (%u elements, %zu bytes)\n",
           buffer_mb, num_elements, bytes);
    fprintf(stderr, "Flops/byte      : %.1f\n", flops_per_byte);
    fprintf(stderr, "Threads/block   : %d\n", threads_per_block);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    /* Load the Metal shader library from source. */
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

    /* Copy data to Metal buffer. */
    memcpy(d_data.contents, h_data, bytes);

    /* Run benchmark at the specified flops_per_byte ratio. */
    double avg_time_ms = 0.0;
    double achieved_gflops = 0.0;
    double achieved_bw = 0.0;

    benchmark_ratio(device, pso, queue, d_data, num_elements, flops_per_byte,
                    threads_per_block,
                    num_warmup,
                    &avg_time_ms, &achieved_gflops, &achieved_bw);

    fprintf(stderr, "FLOP/byte=%.1f  Time=%.3f ms  GFLOPS=%.2f  BW=%.2f GB/s\n",
            flops_per_byte, avg_time_ms, achieved_gflops, achieved_bw);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"mixbench_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"buffer_size_mb\":%d,\"flops_per_byte\":%.1f,"
           "\"threads_per_block\":%d}}\n",
           avg_time_ms, buffer_mb, flops_per_byte, threads_per_block);

    /* JSON-lines output to stdout: summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f},"
           "{\"name\":\"bandwidth_gbps\",\"value\":%.2f}]}\n",
           avg_time_ms, achieved_gflops, achieved_bw);

    /* Clean up. */
    free(h_data);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
