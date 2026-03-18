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
 * Timing is performed using MTLCommandBuffer addCompletedHandler and
 * mach_absolute_time for wall-clock accuracy on Apple Silicon.
 *
 * Usage:
 *   ./memory_bandwidth [buffer_size_mb]
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
// Benchmark H2D
// ---------------------------------------------------------------------------

/**
 * Benchmark host-to-device bandwidth.
 *
 * Allocates a shared (managed) source buffer and a private device destination
 * buffer.  Each iteration blits the full contents from host to device.
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

    // Initialize source data
    memset(src.contents, 0xAB, bytes);

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
// Benchmark D2H
// ---------------------------------------------------------------------------

/**
 * Benchmark device-to-host bandwidth.
 *
 * Allocates a private device source buffer and a shared destination buffer.
 * Each iteration blits from GPU to CPU-accessible memory.
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
// Benchmark D2D
// ---------------------------------------------------------------------------

/**
 * Benchmark device-to-device bandwidth.
 *
 * Uses a Metal compute kernel (d2d_copy from memory_bandwidth.metal) that
 * loads and stores float4 elements to saturate memory bandwidth.
 *
 * @param device    MTLDevice to use
 * @param library   Pre-compiled MTLLibrary containing d2d_copy kernel
 * @param bytes     Transfer size in bytes (must be a multiple of 16)
 * @return          Average bandwidth in GB/s
 */
static double benchmark_d2d(id<MTLDevice> device,
                             id<MTLLibrary> library,
                             size_t bytes)
{
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Build pipeline state for the d2d_copy kernel
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

    // Buffers (both private for maximum bandwidth)
    id<MTLBuffer> src = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModePrivate];

    // Seed src with data via a shared staging buffer
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

    // Number of float4 elements (16 bytes each)
    uint32_t num_f4 = (uint32_t)(bytes / sizeof(float) / 4);

    // Thread-group configuration
    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > 256) tg_size = 256;
    NSUInteger grid_size = (num_f4 + tg_size - 1) / tg_size * tg_size;

    // Warm-up
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

    // Timed iterations
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
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
    // D2D touches both src (read) and dst (write) — 2× bytes per iteration
    return ((double)bytes * 2.0 * NUM_ITERATIONS) / elapsed_s / BYTES_PER_GB;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    @autoreleasepool {

    // Parse optional buffer size argument
    int buffer_mb = DEFAULT_BUFFER_MB;
    if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
        if (buffer_mb <= 0) {
            fprintf(stderr, "Invalid buffer size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    // Align to float4 (16 bytes)
    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;
    bytes = (bytes + 15) & ~(size_t)15;   // round up to 16-byte boundary

    // Select GPU device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("Device: %s\n", device.name.UTF8String);
    printf("Transfer size: %d MiB (%zu bytes)\n", buffer_mb, bytes);
    printf("Iterations per test: %d\n\n", NUM_ITERATIONS);

    // Compile the Metal shader library at runtime from source file
    NSError *err = nil;

    // Try loading pre-compiled .metallib first, then fall back to source
    id<MTLLibrary> library = nil;

    // Look for the .metal source next to the executable
    NSString *metalSrc = nil;
    {
        // Path relative to the executable
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"memory_bandwidth.metal"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:nil];
        }
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
        // Fall back to the default library (empty — just skip D2D)
        fprintf(stderr,
                "Warning: memory_bandwidth.metal not found next to binary; "
                "D2D test will be skipped.\n");
    }

    // ----- H2D -----
    double h2d_bw = benchmark_h2d(device, bytes);
    printf("H2D bandwidth : %8.2f GB/s\n", h2d_bw);

    // ----- D2H -----
    double d2h_bw = benchmark_d2h(device, bytes);
    printf("D2H bandwidth : %8.2f GB/s\n", d2h_bw);

    // ----- D2D -----
    if (library) {
        double d2d_bw = benchmark_d2d(device, library, bytes);
        printf("D2D bandwidth : %8.2f GB/s\n", d2d_bw);
    } else {
        printf("D2D bandwidth : (skipped — .metal source not found)\n");
    }

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
