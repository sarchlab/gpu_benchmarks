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
 * Usage:
 *   ./warp_shuffle [ops_per_thread]
 *
 *   ops_per_thread  – optional number of shuffle iterations per thread
 *                     (default 4096).
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_ops_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_shuffle_variant   ("all", "shfl_xor", "shfl_up", "shfl_down", "shfl")
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

static const int DEFAULT_OPS_PER_THREAD  = 4096;
static const int DEFAULT_NUM_BLOCKS      = 1024;
static const int DEFAULT_THREADS_PER_BLK = 256;
static const int SIMD_WIDTH              = 32;
static const int DEFAULT_NUM_WARMUP      = 0;

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

static int run_benchmark(const char *kernel_name,
                         const char *fn_name,
                         id<MTLDevice> device,
                         id<MTLLibrary> library,
                         id<MTLCommandQueue> queue,
                         id<MTLBuffer> d_out,
                         int ops_per_thread, int num_blocks,
                         int threads_per_block, const char *shuffle_variant,
                         int num_warmup,
                         double *out_avg_ms, double *out_gops)
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
    NSUInteger tg_size = (NSUInteger)threads_per_block;
    if (tg_size > pso.maxTotalThreadsPerThreadgroup)
        tg_size = pso.maxTotalThreadsPerThreadgroup;

    int total_threads = num_blocks * threads_per_block;
    uint32_t opt_u32 = (uint32_t)ops_per_thread;

    /* Helper lambda: dispatch one kernel invocation. */
    auto dispatch_once = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:d_out offset:0 atIndex:0];
        [enc setBytes:&opt_u32 length:sizeof(opt_u32) atIndex:1];
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

    double elapsed_s  = ticks_to_seconds(t1 - t0);
    double elapsed_ms = elapsed_s * 1e3;

    long long num_threads_ll = (long long)total_threads;
    long long total_ops   = num_threads_ll
                            * (long long)ops_per_thread;
    double gops            = (double)total_ops / elapsed_s / 1e9;

    

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f Gops/s\n",
            kernel_name, elapsed_ms, avg_ms, gops);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"ops_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d,\"shuffle_variant\":\"%s\"}}\n",
           kernel_name, (double)elapsed_ms, ops_per_thread, num_blocks,
           threads_per_block, shuffle_variant);

    *out_avg_ms = (double)elapsed_ms;
    *out_gops   = gops;

    return EXIT_SUCCESS;
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
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_ops_per_thread");
    if (env_val) {
        ops_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
    }
    if (ops_per_thread <= 0) {
        fprintf(stderr, "Invalid ops_per_thread\n");
        return EXIT_FAILURE;
    }

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    const char* shuffle_variant = "all";
    env_val = getenv("BENCH_PARAM_shuffle_variant");
    if (env_val) shuffle_variant = env_val;

    int num_warmup = DEFAULT_NUM_WARMUP;
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) num_warmup = atoi(env_val);

    /* Select the default Metal GPU device. */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    int total_threads = num_blocks * threads_per_block;

    fprintf(stderr, "Device          : %s\n", device.name.UTF8String);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
            num_blocks, threads_per_block, total_threads);
    fprintf(stderr, "SIMD width      : %d\n", SIMD_WIDTH);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Shuffle variant : %s\n", shuffle_variant);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

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

    double total_time_ms = 0.0;
    double max_gops = 0.0;

    bool run_xor   = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_xor") == 0);
    bool run_basic = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl") == 0);
    bool run_up    = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_up") == 0);
    bool run_down  = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "shfl_down") == 0);

    int rc;

    if (run_xor) {
        double avg_ms, gops;
        rc = run_benchmark("shfl_xor_kernel", "shfl_xor_kern",
                           device, library, queue, d_out,
                           ops_per_thread, num_blocks, threads_per_block,
                           shuffle_variant, num_warmup, &avg_ms, &gops);
        if (rc != EXIT_SUCCESS) return rc;
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_basic) {
        double avg_ms, gops;
        rc = run_benchmark("shfl_kernel", "shfl_kern",
                           device, library, queue, d_out,
                           ops_per_thread, num_blocks, threads_per_block,
                           shuffle_variant, num_warmup, &avg_ms, &gops);
        if (rc != EXIT_SUCCESS) return rc;
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_up) {
        double avg_ms, gops;
        rc = run_benchmark("shfl_up_kernel", "shfl_up_kern",
                           device, library, queue, d_out,
                           ops_per_thread, num_blocks, threads_per_block,
                           shuffle_variant, num_warmup, &avg_ms, &gops);
        if (rc != EXIT_SUCCESS) return rc;
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_down) {
        double avg_ms, gops;
        rc = run_benchmark("shfl_down_kernel", "shfl_down_kern",
                           device, library, queue, d_out,
                           ops_per_thread, num_blocks, threads_per_block,
                           shuffle_variant, num_warmup, &avg_ms, &gops);
        if (rc != EXIT_SUCCESS) return rc;
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"giga_ops_per_sec\",\"value\":%.2f}]}\n",
           total_time_ms, max_gops);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
