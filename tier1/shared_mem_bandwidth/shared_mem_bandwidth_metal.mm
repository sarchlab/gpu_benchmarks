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
 * Environment variables (override defaults):
 *   BENCH_PARAM_inner_iters
 *   BENCH_PARAM_block_size
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_access_pattern   ("no_conflict", "conflict", or "both")
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

static const int DEFAULT_BLOCK_SIZE   = 256;
static const int DEFAULT_NUM_GROUPS   = 128;
static const int DEFAULT_INNER_ITERS  = 1000;
static const int DEFAULT_WARMUP_ITERS  = 0;
static const int TIMED_ITERS          = 5;
static const int SMEM_FLOATS          = 12288;         // 48 KB / 4 bytes
static const double BYTES_PER_GB      = 1e9;

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
 */
static double run_benchmark(id<MTLDevice>  device,
                            id<MTLLibrary> library,
                            NSString      *kernelName,
                            int            block_size,
                            int            num_groups,
                            int            inner_iters,
                            int            num_warmup,
                            double        *out_bw_GBs)
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
        [device newBufferWithLength:num_groups * sizeof(float)
                            options:MTLResourceStorageModeShared];

    Params params;
    params.inner_iters     = inner_iters;
    params.smem_floats     = SMEM_FLOATS;
    params.threads_per_group = block_size;

    NSUInteger smem_bytes = SMEM_FLOATS * sizeof(float);

    NSUInteger tg_size  = block_size;
    NSUInteger grid_threads = (NSUInteger)num_groups * tg_size;

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
    for (int i = 0; i < num_warmup; ++i)
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
                              * (double)inner_iters * num_groups;
    double bw_GBs = bytes_per_launch / (avg_ms * 1e-3) / BYTES_PER_GB;

    if (out_bw_GBs) *out_bw_GBs = bw_GBs;
    return avg_ms;
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

    /* Read parameters from env vars with defaults. */
    int inner_iters = DEFAULT_INNER_ITERS;
    env_val = getenv("BENCH_PARAM_inner_iters");
    if (env_val) inner_iters = atoi(env_val);

    int block_size = DEFAULT_BLOCK_SIZE;
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int num_groups = DEFAULT_NUM_GROUPS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_groups = atoi(env_val);

    const char* access_pattern = "both";
    env_val = getenv("BENCH_PARAM_access_pattern");
    if (env_val) access_pattern = env_val;

    int num_warmup = DEFAULT_WARMUP_ITERS;
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    // Select GPU device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device        : %s\n", device.name.UTF8String);
    fprintf(stderr, "Block size    : %d threads\n", block_size);
    fprintf(stderr, "Grid size     : %d groups\n", num_groups);
    fprintf(stderr, "Shared mem    : %d bytes (%d floats)\n",
           SMEM_FLOATS * (int)sizeof(float), SMEM_FLOATS);
    fprintf(stderr, "Inner iters   : %d\n", inner_iters);
    fprintf(stderr, "Access pattern: %s\n", access_pattern);
    fprintf(stderr, "Warmup iters  : %d\n", num_warmup);
    fprintf(stderr, "Timed iters   : %d\n\n", TIMED_ITERS);

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

    double total_time_ms = 0.0;
    double max_bw = 0.0;

    bool run_no_conflict = (strcmp(access_pattern, "no_conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);
    bool run_conflict    = (strcmp(access_pattern, "conflict") == 0 ||
                            strcmp(access_pattern, "both") == 0);

    // --- No-conflict ---
    if (run_no_conflict) {
        double nc_bw = 0.0;
        double nc_ms = run_benchmark(device, library,
                                     @"smem_bw_no_conflict",
                                     block_size, num_groups, inner_iters,
                                     num_warmup, &nc_bw);
        fprintf(stderr, "no_conflict    : %.3f ms  %.2f GB/s\n", nc_ms, nc_bw);
        total_time_ms += nc_ms;
        if (nc_bw > max_bw) max_bw = nc_bw;

        printf("{\"type\":\"kernel\",\"name\":\"smem_bw_no_conflict\",\"time_ms\":%.6f,"
               "\"params\":{\"inner_iters\":%d,\"block_size\":%d,"
               "\"num_blocks\":%d,\"access_pattern\":\"%s\"}}\n",
               nc_ms, inner_iters, block_size, num_groups, access_pattern);
    }

    // --- Conflict ---
    if (run_conflict) {
        double c_bw = 0.0;
        double c_ms = run_benchmark(device, library,
                                    @"smem_bw_conflict",
                                    block_size, num_groups, inner_iters,
                                    num_warmup, &c_bw);
        fprintf(stderr, "conflict       : %.3f ms  %.2f GB/s\n", c_ms, c_bw);
        total_time_ms += c_ms;
        if (c_bw > max_bw) max_bw = c_bw;

        printf("{\"type\":\"kernel\",\"name\":\"smem_bw_conflict\",\"time_ms\":%.6f,"
               "\"params\":{\"inner_iters\":%d,\"block_size\":%d,"
               "\"num_blocks\":%d,\"access_pattern\":\"%s\"}}\n",
               c_ms, inner_iters, block_size, num_groups, access_pattern);
    }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"bandwidth_gbps\",\"value\":%.2f}]}\n",
           total_time_ms, max_bw);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
