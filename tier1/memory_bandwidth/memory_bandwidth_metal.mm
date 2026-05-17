/**
 * memory_bandwidth_metal.mm
 *
 * Apple Metal port of the memory_bandwidth microbenchmark.
 *
 * Measures GPU memory bandwidth in three directions:
 *   - Host-to-Device (H2D): CPU → GPU via MTLBuffer blit
 *   - Device-to-Host (D2H): GPU → CPU via MTLBuffer blit
 *   - Device-to-Device (D2D): GPU → GPU via Metal compute kernel (d2d_copy)
 *
 * Timing is performed using mach_absolute_time for wall-clock accuracy
 * on Apple Silicon.
 *
 * Usage:
 *   ./memory_bandwidth [buffer_size_mb]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_transfer_type     (h2d, d2h, d2d, all)
 *   BENCH_PARAM_use_pinned_memory (true, false)
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

static const int    DEFAULT_BUFFER_MB      = 256;
static const int    DEFAULT_NUM_WARMUP     = 0;
static const double BYTES_PER_GB           = 1e9;

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

struct BenchResult {
    double bandwidth_gbps;
    double total_ms;
};

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
// Benchmark H2D
// ---------------------------------------------------------------------------

static BenchResult benchmark_h2d(id<MTLDevice> device, size_t bytes,
                                  int num_warmup)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> src = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];
    memset(src.contents, 0xAB, bytes);

    /* Warm-up. */
    for (int w = 0; w < num_warmup; ++w) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0
                    toBuffer:dst destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint64_t t0 = mach_absolute_time();
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0
                    toBuffer:dst destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    double bw = (double)bytes / elapsed_s / BYTES_PER_GB;
    return {bw, elapsed_s * 1e3};
}

// ---------------------------------------------------------------------------
// Benchmark D2H
// ---------------------------------------------------------------------------

static BenchResult benchmark_d2h(id<MTLDevice> device, size_t bytes,
                                  int num_warmup)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> src = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];

    /* Seed the private source via a shared staging buffer. */
    {
        id<MTLBuffer> stage = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        memset(stage.contents, 0xCD, bytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:stage sourceOffset:0
                    toBuffer:src destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /* Warm-up. */
    for (int w = 0; w < num_warmup; ++w) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0
                    toBuffer:dst destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint64_t t0 = mach_absolute_time();
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0
                    toBuffer:dst destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    double bw = (double)bytes / elapsed_s / BYTES_PER_GB;
    return {bw, elapsed_s * 1e3};
}

// ---------------------------------------------------------------------------
// Benchmark D2D
// ---------------------------------------------------------------------------

static BenchResult benchmark_d2d(id<MTLDevice> device,
                                  id<MTLLibrary> library,
                                  size_t bytes,
                                  int num_warmup)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError *err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@"d2d_copy"];
    if (!fn) {
        fprintf(stderr, "Error: could not find kernel 'd2d_copy' in library\n");
        exit(EXIT_FAILURE);
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline state: %s\n",
                err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }

    id<MTLBuffer> src = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];

    /* Seed src with data. */
    {
        id<MTLBuffer> stage = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        memset(stage.contents, 0xEF, bytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:stage sourceOffset:0
                    toBuffer:src destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint32_t num_f4 = (uint32_t)(bytes / sizeof(float) / 4);
    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > 256) tg_size = 256;
    NSUInteger grid_size = (num_f4 + tg_size - 1) / tg_size * tg_size;

    /* Warm-up. */
    for (int w = 0; w < num_warmup; ++w) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&num_f4 length:sizeof(num_f4) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint64_t t0 = mach_absolute_time();
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&num_f4 length:sizeof(num_f4) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    /* D2D touches both src (read) and dst (write) — 2× bytes. */
    double bw = ((double)bytes * 2.0) / elapsed_s / BYTES_PER_GB;
    return {bw, elapsed_s * 1e3};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    int buffer_mb = DEFAULT_BUFFER_MB;
    env_val = getenv("BENCH_PARAM_buffer_size_mb");
    if (env_val) {
        buffer_mb = atoi(env_val);
    } else if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
    }
    if (buffer_mb <= 0) {
        fprintf(stderr, "Invalid buffer size\n");
        return EXIT_FAILURE;
    }


    const char* transfer_type = "all";
    env_val = getenv("BENCH_PARAM_transfer_type");
    if (env_val) transfer_type = env_val;

    const char* use_pinned_memory = "true";
    env_val = getenv("BENCH_PARAM_use_pinned_memory");
    if (env_val) use_pinned_memory = env_val;

    int num_warmup = DEFAULT_NUM_WARMUP;

    bool run_h2d = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "h2d") == 0);
    bool run_d2h = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2h") == 0);
    bool run_d2d = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2d") == 0);

    /* Align to float4 (16 bytes). */
    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;
    bytes = (bytes + 15) & ~(size_t)15;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Transfer size: %d MiB (%zu bytes)\n", buffer_mb, bytes);
    fprintf(stderr, "Transfer type: %s\n", transfer_type);
    fprintf(stderr, "Pinned memory: %s\n", use_pinned_memory);
    fprintf(stderr, "Warmup iters: %d\n\n", num_warmup);

    /* Compile the Metal shader library at runtime (needed for D2D kernel). */
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"memory_bandwidth.metal"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            NSString *metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil];
            if (metalSrc) {
                MTLCompileOptions *opts = [MTLCompileOptions new];
                library = [device newLibraryWithSource:metalSrc options:opts error:&err];
                if (!library) {
                    fprintf(stderr, "Metal compile error: %s\n",
                            err.localizedDescription.UTF8String);
                    return EXIT_FAILURE;
                }
            }
        }
    }

    if (!library && run_d2d) {
        fprintf(stderr,
                "Warning: memory_bandwidth.metal not found next to binary; "
                "D2D test will be skipped.\n");
        run_d2d = false;
    }

    double total_time_ms = 0.0;
    double h2d_bw = 0.0, d2h_bw = 0.0, d2d_bw = 0.0;

    /* ----- H2D ----- */
    if (run_h2d) {
        BenchResult r = benchmark_h2d(device, bytes, num_warmup);
        h2d_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, "H2D bandwidth : %8.2f GB/s\n", h2d_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_h2d\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- D2H ----- */
    if (run_d2h) {
        BenchResult r = benchmark_d2h(device, bytes, num_warmup);
        d2h_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, "D2H bandwidth : %8.2f GB/s\n", d2h_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_d2h\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- D2D ----- */
    if (run_d2d) {
        BenchResult r = benchmark_d2d(device, library, bytes, num_warmup);
        d2d_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, "D2D bandwidth : %8.2f GB/s\n", d2d_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_d2d\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* Summary. */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,\"metrics\":[", total_time_ms);
    bool first_metric = true;
    if (run_h2d) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"h2d_bandwidth_gbps\",\"value\":%.2f}", h2d_bw);
        first_metric = false;
    }
    if (run_d2h) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"d2h_bandwidth_gbps\",\"value\":%.2f}", d2h_bw);
        first_metric = false;
    }
    if (run_d2d) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"d2d_bandwidth_gbps\",\"value\":%.2f}", d2d_bw);
        first_metric = false;
    }
    printf("]}\n");

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
