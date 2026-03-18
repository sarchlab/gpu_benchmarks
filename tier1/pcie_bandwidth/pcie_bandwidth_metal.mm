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
 * This benchmark provides an equivalent metric for comparing host↔GPU
 * transfer latency and bandwidth across platforms (CUDA/HIP/Metal).
 *
 * Timing uses mach_absolute_time() for wall-clock accuracy.
 * Command buffers are committed one at a time and waited on
 * (waitUntilCompleted) to give a faithful per-transfer measurement.
 *
 * Usage:
 *   ./pcie_bandwidth [buffer_size_mb]
 *
 *   buffer_size_mb  – optional transfer size in MiB (default 256)
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

static const int    DEFAULT_BUFFER_MB = 256;
static const int    NUM_ITERATIONS    = 20;
static const double BYTES_PER_GB      = 1e9;

// ---------------------------------------------------------------------------
// Timing helpers
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
// Benchmark: unidirectional Host → Device
// ---------------------------------------------------------------------------

/**
 * Measure H2D (shared → private) bandwidth via MTLBlitCommandEncoder.
 *
 * Allocates a CPU-visible shared buffer (source) and a GPU-only private
 * buffer (destination).  Each iteration blits the full contents from
 * shared to private memory.
 *
 * On Apple Silicon the "H2D" transfer moves data from the CPU-coherent
 * region of unified memory to the GPU-tiled (private) region.
 *
 * @param device    MTLDevice to use
 * @param bytes     Transfer size in bytes
 * @return          Average bandwidth in GB/s
 */
static double benchmark_h2d(id<MTLDevice> device, size_t bytes)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Source: shared buffer (CPU-accessible)
    id<MTLBuffer> src = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    // Destination: private buffer (GPU-only)
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];

    // Initialize source with a recognizable pattern
    memset(src.contents, 0xAB, bytes);

    // Warm-up: one blit to prime the memory fabric
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

    // Timed iterations
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
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
    return ((double)bytes * NUM_ITERATIONS) / elapsed_s / BYTES_PER_GB;
}

// ---------------------------------------------------------------------------
// Benchmark: unidirectional Device → Host
// ---------------------------------------------------------------------------

/**
 * Measure D2H (private → shared) bandwidth via MTLBlitCommandEncoder.
 *
 * Allocates a GPU-only private buffer (source) and a CPU-visible shared
 * buffer (destination).  A staging blit seeds the private source before
 * timing begins.
 *
 * @param device    MTLDevice to use
 * @param bytes     Transfer size in bytes
 * @return          Average bandwidth in GB/s
 */
static double benchmark_d2h(id<MTLDevice> device, size_t bytes)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Source: private buffer (GPU-only)
    id<MTLBuffer> src = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];
    // Destination: shared buffer (CPU-accessible)
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];

    // Seed the private source via a shared staging buffer
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

    // Warm-up
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

    // Timed iterations
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
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
    return ((double)bytes * NUM_ITERATIONS) / elapsed_s / BYTES_PER_GB;
}

// ---------------------------------------------------------------------------
// Benchmark: bidirectional (interleaved H2D + D2H in one command buffer)
// ---------------------------------------------------------------------------

/**
 * Measure aggregate bidirectional bandwidth.
 *
 * Each iteration encodes both an H2D blit (shared→private) and a D2H blit
 * (private→shared) into the same MTLCommandBuffer.  The Metal runtime
 * may overlap them on the blit engine.  The reported aggregate bandwidth
 * is (2 × bytes × iterations) / elapsed_time.
 *
 * @param device    MTLDevice to use
 * @param bytes     Transfer size per direction in bytes
 * @return          Aggregate bidirectional bandwidth in GB/s
 */
static double benchmark_bidir(id<MTLDevice> device, size_t bytes)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // H2D buffers: shared source → private destination
    id<MTLBuffer> h_src = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> d_h2d = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModePrivate];

    // D2H buffers: private source → shared destination
    id<MTLBuffer> d_src = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModePrivate];
    id<MTLBuffer> h_dst = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];

    // Initialize shared source
    memset(h_src.contents, 0xAB, bytes);

    // Seed d_src via staging blit
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

    // Warm-up: one combined blit (H2D + D2H in a single command buffer)
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

    // Timed iterations
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
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
    // Each iteration moves bytes in each direction → 2*bytes total
    return (2.0 * (double)bytes * NUM_ITERATIONS) / elapsed_s / BYTES_PER_GB;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    // Parse optional buffer size argument (in MiB)
    int buffer_mb = DEFAULT_BUFFER_MB;
    if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
        if (buffer_mb <= 0) {
            fprintf(stderr, "Invalid buffer size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;

    // Select GPU device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("Device          : %s\n", device.name.UTF8String);
    printf("Transfer size   : %d MiB (%zu bytes)\n", buffer_mb, bytes);
    printf("Iterations/test : %d\n", NUM_ITERATIONS);
    printf("Note            : Apple Silicon uses unified memory; H2D/D2H\n");
    printf("                  measures shared↔private memory fabric copy overhead.\n\n");

    printf("%-30s", "H2D (unidirectional)");
    fflush(stdout);
    double h2d_bw = benchmark_h2d(device, bytes);
    printf(": %8.2f GB/s\n", h2d_bw);

    printf("%-30s", "D2H (unidirectional)");
    fflush(stdout);
    double d2h_bw = benchmark_d2h(device, bytes);
    printf(": %8.2f GB/s\n", d2h_bw);

    printf("%-30s", "Bidirectional (H2D + D2H)");
    fflush(stdout);
    double bidir_bw = benchmark_bidir(device, bytes);
    printf(": %8.2f GB/s  (aggregate)\n", bidir_bw);

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
