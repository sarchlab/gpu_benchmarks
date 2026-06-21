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
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./int32_throughput [mads_per_thread]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_mads_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
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
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_MADS_PER_THREAD = 4096;
static const int DEFAULT_NUM_BLOCKS        = 1024;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_WARMUP = 5;

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

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

    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    int mads_per_thread = DEFAULT_MADS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_mads_per_thread");
    if (env_val) {
        mads_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        mads_per_thread = atoi(argv[1]);
    }
    if (mads_per_thread <= 0) {
        fprintf(stderr, "Invalid mads_per_thread\n");
        return EXIT_FAILURE;
    }
    mads_per_thread = (mads_per_thread / 4) * 4;
    if (mads_per_thread == 0) mads_per_thread = 4;

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    int num_warmup = DEFAULT_NUM_WARMUP;

    int total_threads = num_blocks * threads_per_block;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device          : %s\n", device.name.UTF8String);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "MADs / thread   : %d\n", mads_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

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

    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> d_out = [device newBufferWithLength:sizeof(int)
                                             options:MTLResourceStorageModeShared];

    NSUInteger tg_size = (NSUInteger)threads_per_block;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;

    uint32_t mpt_u32 = (uint32_t)mads_per_thread;

    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_out    offset:0 atIndex:0];
        [enc setBytes:&mpt_u32  length:sizeof(mpt_u32) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total_threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
        dispatch_once();
    }

    /* Timed kernel launch. */
    uint64_t t0 = mach_absolute_time();
    dispatch_once();
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    long long num_threads = (long long)total_threads;
    long long total_ops   = num_threads
                            * (long long)mads_per_thread
                            * 2LL;
    double giops = (double)total_ops / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms\n", elapsed_ms);
    fprintf(stderr, "Total INT ops   : %lld\n", total_ops);
    fprintf(stderr, "INT32 Throughput : %.2f GIOPS\n", giops);

    /* JSON-lines output to stdout. */
    printf("{\"type\":\"kernel\",\"name\":\"int32_mad_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"mads_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d}}\n",
           (double)elapsed_ms, mads_per_thread, num_blocks, threads_per_block);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GIOPS\",\"value\":%.2f}]}\n",
           elapsed_ms, giops);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
