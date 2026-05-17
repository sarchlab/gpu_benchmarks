/**
 * pcie_bandwidth_metal.mm
 *
 * Apple Metal port of the pcie_bandwidth microbenchmark.
 *
 * Measures host↔GPU transfer bandwidth in three modes:
 *   1. Unidirectional H2D — shared (CPU-visible) buffer → private (GPU-only)
 *      buffer via MTLBlitCommandEncoder, reports GB/s
 *   2. Unidirectional D2H — private (GPU-only) buffer → shared (CPU-visible)
 *      buffer via MTLBlitCommandEncoder, reports GB/s
 *   3. Bidirectional       — interleaved H2D and D2H blits in a single
 *      command buffer, reports aggregate (sum) GB/s
 *
 * NOTE: Apple Silicon uses a unified memory architecture — there is no
 * discrete PCIe bus between CPU and GPU.  Both processors share the same
 * physical memory fabric.  MTLResourceStorageModeShared buffers are
 * accessible by both CPU and GPU without any copying.
 * MTLResourceStorageModePrivate buffers reside in GPU-optimized tiled memory;
 * blitting between Shared and Private measures the internal memory-fabric
 * copy overhead rather than a PCIe transfer.
 *
 * Timing uses mach_absolute_time() for wall-clock accuracy.
 *
 * Usage:
 *   ./pcie_bandwidth [buffer_size_mb]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_transfer_type     (h2d, d2h, bidir, all)
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
// Benchmark: unidirectional Host → Device
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
// Benchmark: unidirectional Device → Host
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
// Benchmark: bidirectional (interleaved H2D + D2H)
// ---------------------------------------------------------------------------

static BenchResult benchmark_bidir(id<MTLDevice> device, size_t bytes,
                                    int num_warmup)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* H2D buffers: shared source → private destination. */
    id<MTLBuffer> h_src = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> d_h2d = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModePrivate];

    /* D2H buffers: private source → shared destination. */
    id<MTLBuffer> d_src = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModePrivate];
    id<MTLBuffer> h_dst = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];

    memset(h_src.contents, 0xAB, bytes);

    /* Seed d_src via staging blit. */
    {
        id<MTLBuffer> stage = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        memset(stage.contents, 0xEF, bytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:stage sourceOffset:0
                    toBuffer:d_src destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /* Warm-up. */
    for (int w = 0; w < num_warmup; ++w) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:h_src sourceOffset:0
                    toBuffer:d_h2d destinationOffset:0
                        size:bytes];
        [blit copyFromBuffer:d_src sourceOffset:0
                    toBuffer:h_dst destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint64_t t0 = mach_absolute_time();
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:h_src sourceOffset:0
                    toBuffer:d_h2d destinationOffset:0
                        size:bytes];
        [blit copyFromBuffer:d_src sourceOffset:0
                    toBuffer:h_dst destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    uint64_t t1 = mach_absolute_time();

    double elapsed_s = ticks_to_seconds(t1 - t0);
    /* Moves bytes in each direction → 2*bytes total. */
    double bw = (2.0 * (double)bytes) / elapsed_s / BYTES_PER_GB;
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

    bool run_h2d   = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "h2d") == 0);
    bool run_d2h   = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2h") == 0);
    bool run_bidir = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "bidir") == 0);

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Device          : %s\n", device.name.UTF8String);
    fprintf(stderr, "Transfer size   : %d MiB (%zu bytes)\n", buffer_mb, bytes);
    fprintf(stderr, "Transfer type   : %s\n", transfer_type);
    fprintf(stderr, "Pinned memory   : %s\n", use_pinned_memory);
    fprintf(stderr, "Warmup iters    : %d\n", num_warmup);
    fprintf(stderr, "Note            : Apple Silicon uses unified memory; H2D/D2H\n");
    fprintf(stderr, "                  measures shared↔private memory fabric copy overhead.\n\n");

    double total_time_ms = 0.0;
    double h2d_bw = 0.0, d2h_bw = 0.0, bidir_bw = 0.0;

    /* ----- H2D ----- */
    if (run_h2d) {
        fprintf(stderr, "%-30s", "H2D (unidirectional)");
        BenchResult r = benchmark_h2d(device, bytes, num_warmup);
        h2d_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, ": %8.2f GB/s\n", h2d_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_h2d\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- D2H ----- */
    if (run_d2h) {
        fprintf(stderr, "%-30s", "D2H (unidirectional)");
        BenchResult r = benchmark_d2h(device, bytes, num_warmup);
        d2h_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, ": %8.2f GB/s\n", d2h_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_d2h\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- Bidirectional ----- */
    if (run_bidir) {
        fprintf(stderr, "%-30s", "Bidirectional (H2D + D2H)");
        BenchResult r = benchmark_bidir(device, bytes, num_warmup);
        bidir_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, ": %8.2f GB/s  (aggregate)\n", bidir_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_bidir\",\"time_ms\":%.6f,"
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
    if (run_bidir) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"bidir_bandwidth_gbps\",\"value\":%.2f}", bidir_bw);
        first_metric = false;
    }
    printf("]}\n");

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
