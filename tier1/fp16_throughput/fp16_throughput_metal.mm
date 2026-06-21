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
 * Environment variables (override argv):
 *   BENCH_PARAM_fmas_per_thread
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

static const int DEFAULT_FMAS_PER_THREAD = 4096;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_BLOCKS = 1024;
static const int DEFAULT_NUM_WARMUP = 5;

// ---------------------------------------------------------------------------
// Timing helpers
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

    int fmas_per_thread = DEFAULT_FMAS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_fmas_per_thread");
    if (env_val) {
        fmas_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        fmas_per_thread = atoi(argv[1]);
    }
    if (fmas_per_thread <= 0) {
        fprintf(stderr, "Invalid fmas_per_thread\n");
        return EXIT_FAILURE;
    }
    fmas_per_thread = (fmas_per_thread / 4) * 4;
    if (fmas_per_thread == 0) fmas_per_thread = 4;

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
    fprintf(stderr, "FMAs / thread   : %d  (fma(half2,...), 4 FLOPs each)\n",
           fmas_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    NSError *err = nil;
    id<MTLLibrary> library = nil;

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

    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> outBuf = [device newBufferWithLength:4
                                              options:MTLResourceStorageModeShared];

    NSUInteger tg_size  = (NSUInteger)threads_per_block;
    NSUInteger grid_dim = (NSUInteger)total_threads;

    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
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

    /* Timed kernel launch. */
    uint64_t t0 = mach_absolute_time();
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
    uint64_t t1 = mach_absolute_time();

    long long num_threads_ll = (long long)total_threads;
    long long total_flops = num_threads_ll
                            * (long long)fmas_per_thread
                            * 4LL;
    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;
    double gflops     = (double)total_flops / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms\n", elapsed_ms);
    fprintf(stderr, "Total FLOPs     : %lld\n", total_flops);
    fprintf(stderr, "FP16 Throughput : %.2f GFLOPS\n", gflops);

    /* JSON-lines output to stdout. */
    printf("{\"type\":\"kernel\",\"name\":\"fp16_fma_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"fmas_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d}}\n",
           (double)elapsed_ms, fmas_per_thread, num_blocks, threads_per_block);

    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           elapsed_ms, gflops);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
