/**
 * shoc_devicememory_metal.mm — Apple Metal host for SHOC DeviceMemory benchmark.
 *
 * Measures device global memory bandwidth for three access patterns:
 *   - Read-only
 *   - Write-only
 *   - Read-write
 *
 * Usage:
 *   ./shoc_devicememory [--array_size_mb N] [--iterations I]
 *
 *   --array_size_mb N  Array size in MB (default: 64)
 *   --iterations I     Timed iterations (default: 20)
 *
 * Output (stdout): 3 CSV rows (one per pattern)
 * Output (stderr): bandwidth in GB/s for each pattern
 *
 * Build:
 *   make PLATFORM=metal
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <functional>
#include <vector>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double ticks_to_seconds(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-9;
}

static double ticks_to_ms(uint64_t ticks)
{
    return ticks_to_seconds(ticks) * 1e3;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static int parseIterations(int argc, char** argv) {
    int iters = 20;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--iterations") == 0) {
            int v = atoi(argv[i + 1]);
            if (v >= 1) iters = v;
            break;
        }
    }
    return iters;
}

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

static void printCSVHeader() {
    printf("kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms\n");
}

static void printCSVRow(const BenchResult& r) {
    printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
           r.kernel_name, r.problem_size, r.iterations,
           r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

// ---------------------------------------------------------------------------
// Benchmark helper: warm up then time N iterations
// ---------------------------------------------------------------------------
static BenchResult runBenchmarkMetal(const char*                  kernel_name,
                                     const char*                  problem_size,
                                     int                          iters,
                                     id<MTLCommandQueue>          queue,
                                     id<MTLComputePipelineState>  pso,
                                     NSUInteger                   num_threads,
                                     NSUInteger                   tg_size,
                                     std::function<void(id<MTLComputeCommandEncoder>)> setup_fn)
{
    auto dispatch_one = [&]() {
        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        setup_fn(enc);
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // Warmup
    dispatch_one();

    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = mach_absolute_time();
        dispatch_one();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
    }

    double sum = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iters; ++i) {
        sum += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;

    BenchResult r;
    r.kernel_name  = kernel_name;
    r.problem_size = problem_size;
    r.iterations   = iters;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;
    return r;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {

    int array_size_mb = parseIntParam(argc, argv, "--array_size_mb", 64);
    int iters         = parseIterations(argc, argv);

    size_t N     = (size_t)array_size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Array size: %d MB (%zu floats)  |  Iterations: %d\n\n",
            array_size_mb, N, iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from source file next to the binary
    // -------------------------------------------------------------------
    NSError *err = nil;
    id<MTLLibrary> library = nil;

    NSString *exePath = [[NSBundle mainBundle] executablePath];
    NSString *dir     = [exePath stringByDeletingLastPathComponent];
    NSString *srcPath = [dir stringByAppendingPathComponent:@"shoc_devicememory.metal"];

    if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
        NSString *metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        library = [device newLibraryWithSource:metalSrc options:opts error:&err];
        if (!library) {
            fprintf(stderr, "Metal compile error: %s\n",
                    err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Error: shoc_devicememory.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    // Build pipelines for each kernel
    auto makePSO = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "Error: kernel '%s' not found\n", name.UTF8String);
            exit(EXIT_FAILURE);
        }
        NSError *e = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&e];
        if (!p) {
            fprintf(stderr, "Error creating pipeline for '%s': %s\n",
                    name.UTF8String, e.localizedDescription.UTF8String);
            exit(EXIT_FAILURE);
        }
        return p;
    };

    id<MTLComputePipelineState> pso_read  = makePSO(@"readOnly_kernel");
    id<MTLComputePipelineState> pso_write = makePSO(@"writeOnly_kernel");
    id<MTLComputePipelineState> pso_rw    = makePSO(@"readWrite_kernel");

    // -------------------------------------------------------------------
    // Allocate buffers
    // -------------------------------------------------------------------
    id<MTLBuffer> buf_input  = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_output = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];

    // Initialize input
    float* h_input = (float*)buf_input.contents;
    for (size_t i = 0; i < N; ++i)
        h_input[i] = (float)(i % 1000) * 0.001f;

    // Thread configuration
    NSUInteger tg_size    = 256;
    NSUInteger num_groups = (N + tg_size - 1) / tg_size;
    if (num_groups > 65535) num_groups = 65535;
    NSUInteger num_threads = num_groups * tg_size;

    // params: {N, totalThreads}
    uint32_t params_ro[2] = { (uint32_t)N, (uint32_t)num_threads };
    uint32_t params_wo[2] = { (uint32_t)N, (uint32_t)num_threads };
    uint32_t params_rw[2] = { (uint32_t)N, (uint32_t)num_threads };

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dMB", array_size_mb);

    printCSVHeader();

    // --- Read-only ---
    BenchResult r_read = runBenchmarkMetal(
        "DeviceMemory_Read", size_str, iters, queue, pso_read, num_threads, tg_size,
        [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_input  offset:0 atIndex:0];
            [enc setBuffer:buf_output offset:0 atIndex:1];
            [enc setBytes:params_ro length:sizeof(params_ro) atIndex:2];
        });
    printCSVRow(r_read);
    double read_bw = (double)bytes / (r_read.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Read-only bandwidth:  %.2f GB/s  (%d MB)\n", read_bw, array_size_mb);

    // --- Write-only ---
    BenchResult r_write = runBenchmarkMetal(
        "DeviceMemory_Write", size_str, iters, queue, pso_write, num_threads, tg_size,
        [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_output offset:0 atIndex:0];
            [enc setBytes:params_wo length:sizeof(params_wo) atIndex:1];
        });
    printCSVRow(r_write);
    double write_bw = (double)bytes / (r_write.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Write-only bandwidth: %.2f GB/s  (%d MB)\n", write_bw, array_size_mb);

    // --- Read-write: re-initialize input buffer ---
    for (size_t i = 0; i < N; ++i)
        h_input[i] = (float)(i % 1000) * 0.001f;

    BenchResult r_rw = runBenchmarkMetal(
        "DeviceMemory_ReadWrite", size_str, iters, queue, pso_rw, num_threads, tg_size,
        [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_input offset:0 atIndex:0];
            [enc setBytes:params_rw length:sizeof(params_rw) atIndex:1];
        });
    printCSVRow(r_rw);
    // Read-write: 2× data (1 read + 1 write)
    double rw_bw = 2.0 * (double)bytes / (r_rw.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Read-write bandwidth: %.2f GB/s  (%d MB)\n", rw_bw, array_size_mb);

    return EXIT_SUCCESS;

    } // @autoreleasepool
}
