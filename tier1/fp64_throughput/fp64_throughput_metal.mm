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
 * Timing:
 *   Elapsed time is measured using mach_absolute_time() wall-clock timing
 *   around serialised command buffer executions (waitUntilCompleted).
 *
 * Usage:
 *   ./fp64_throughput [fmas_per_thread]
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
static const int DEFAULT_NUM_BLOCKS        = 1024;
static const int DEFAULT_THREADS_PER_BLOCK = 256;
static const int DEFAULT_NUM_WARMUP = 0;

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
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    int total_threads = num_blocks * threads_per_block;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device          : %s\n", device.name.UTF8String);
    fprintf(stderr, "NOTE            : Apple GPUs lack native FP64 hardware.\n");
    fprintf(stderr, "                  Measuring FP32 throughput as proxy.\n\n");
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
           num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "FMAs / thread   : %d\n", fmas_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

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

    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> d_out = [device newBufferWithLength:sizeof(float)
                                             options:MTLResourceStorageModeShared];

    NSUInteger tg_size = (NSUInteger)threads_per_block;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;

    uint32_t fpt_u32 = (uint32_t)fmas_per_thread;

    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_out    offset:0 atIndex:0];
        [enc setBytes:&fpt_u32  length:sizeof(fpt_u32) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total_threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    for (int w = 0; w < num_warmup; ++w) {
        dispatch_once();
    }

    /* Timed kernel launch. */
    uint64_t t0 = mach_absolute_time();
    dispatch_once();
    uint64_t t1 = mach_absolute_time();

    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    long long num_threads = (long long)total_threads;
    long long total_flops = num_threads
                            * (long long)fmas_per_thread
                            * 2LL;
    double gflops = (double)total_flops / elapsed_s / 1e9;

    fprintf(stderr, "Elapsed time    : %.3f ms\n", elapsed_ms);
    fprintf(stderr, "Total FLOPs     : %lld\n", total_flops);
    fprintf(stderr, "FP32 Throughput : %.2f GFLOPS  (FP64 proxy — Apple GPUs lack native FP64)\n",
           gflops);

    /* JSON-lines output to stdout. */
    printf("{\"type\":\"kernel\",\"name\":\"fp64_fma_kernel\",\"time_ms\":%.6f,"
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
