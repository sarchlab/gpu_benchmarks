/**
 * shoc_reduction_metal.mm — Apple Metal host for the SHOC Reduction benchmark.
 *
 * Parallel sum reduction: reduces a float array to a single sum using
 * tree-based threadgroup (shared-memory) reduction.  Derived from SHOC.
 *
 * The Metal compute kernel lives in shoc_reduction.metal and is compiled at
 * runtime from source (loaded from the same directory as the binary).
 *
 * Two-pass strategy:
 *   Pass 1: reduce_kernel dispatched over the full input array
 *           → partial_sums[0..num_groups-1]
 *   Pass 2: reduce_kernel dispatched over partial_sums
 *           → result[0]  (if partial array is still large)
 *   Repeat Pass 2 until a single value remains.
 *
 * Usage:
 *   ./shoc_reduction [--size N] [--iterations I]
 *
 *   --size N         Number of float elements (default: 4194304 = 4 M)
 *   --iterations I   Timed kernel launches   (default: 20)
 *
 * Output (stdout): CSV — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
 * Output (stderr): Effective bandwidth in GB/s (2*N*sizeof(float)/time), PASS/FAIL
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
#include <vector>
#include <mach/mach_time.h>

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
// Reduction dispatch helper
//
// Runs multi-pass reduction until a single value remains in d_result[0].
// Returns via the provided staging buffer (read back by caller).
// ---------------------------------------------------------------------------

static id<MTLBuffer> dispatch_reduction(
    id<MTLCommandQueue>          queue,
    id<MTLComputePipelineState>  pso,
    NSUInteger                   tg_size,
    id<MTLBuffer>                buf_input,
    NSUInteger                   N_input,        // elements in buf_input
    id<MTLBuffer>                buf_scratch1,   // pre-allocated, size >= ceil(N_input / (2*tg_size))
    id<MTLBuffer>                buf_scratch2)   // pre-allocated, size >= ceil(scratch1 / (2*tg_size))
{
    // We perform successive passes until one element remains.
    // buf_src  → buf_dst, then swap if another pass is needed.

    id<MTLBuffer> buf_src = buf_input;
    NSUInteger    n_src   = N_input;

    // ping-pong between scratch1 and scratch2
    id<MTLBuffer> scratch[2] = { buf_scratch1, buf_scratch2 };
    int scratch_idx = 0;

    while (n_src > 1) {
        NSUInteger num_groups = (n_src + tg_size * 2 - 1) / (tg_size * 2);
        id<MTLBuffer> buf_dst = scratch[scratch_idx & 1];
        scratch_idx++;

        // Encode one reduction pass
        id<MTLCommandBuffer>       cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_src  offset:0 atIndex:0];
        [enc setBuffer:buf_dst  offset:0 atIndex:1];
        uint32_t n32 = (uint32_t)n_src;
        [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
        // threadgroup memory: tg_size floats
        [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        buf_src = buf_dst;
        n_src   = num_groups;
    }

    return buf_src;   // points to the buffer containing result[0]
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {

    int N     = parseIntParam(argc, argv, "--size", 4194304);
    int iters = parseIterations(argc, argv);

    size_t bytes = (size_t)N * sizeof(float);

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Array size: %d floats (%.1f MiB)\n",
            N, (double)bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Iterations: %d\n\n", iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from source file next to the binary
    // -------------------------------------------------------------------
    NSError   *err     = nil;
    id<MTLLibrary> library = nil;

    NSString *exePath = [[NSBundle mainBundle] executablePath];
    NSString *dir     = [exePath stringByDeletingLastPathComponent];
    NSString *srcPath = [dir stringByAppendingPathComponent:@"shoc_reduction.metal"];

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
        fprintf(stderr, "Error: shoc_reduction.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"reduce_kernel"];
    if (!fn) {
        fprintf(stderr, "Error: kernel 'reduce_kernel' not found in library\n");
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // Choose threadgroup size (cap at 256 for broad compatibility)
    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > 256) tg_size = 256;

    // -------------------------------------------------------------------
    // Allocate buffers
    //   buf_input   — input array (shared, CPU fills once)
    //   buf_s1, s2  — ping-pong scratch buffers for multi-pass reduction
    // The largest scratch needed is ceil(N / (2*tg_size)) per pass.
    // -------------------------------------------------------------------
    NSUInteger num_groups1 = ((NSUInteger)N + tg_size * 2 - 1) / (tg_size * 2);
    NSUInteger num_groups2 = (num_groups1 + tg_size * 2 - 1) / (tg_size * 2);
    if (num_groups2 < 1) num_groups2 = 1;

    id<MTLBuffer> buf_input = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_s1    = [device newBufferWithLength:num_groups1 * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_s2    = [device newBufferWithLength:(num_groups2 < 1 ? 1 : num_groups2) * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

    // Initialize input with known pattern (same as HIP version)
    float* h_input = (float*)buf_input.contents;
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i % 1000) * 0.001f;
    }

    // -------------------------------------------------------------------
    // Warmup pass
    // -------------------------------------------------------------------
    id<MTLBuffer> buf_result = dispatch_reduction(queue, pso, tg_size,
                                                  buf_input, (NSUInteger)N,
                                                  buf_s1, buf_s2);

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = mach_absolute_time();
        buf_result = dispatch_reduction(queue, pso, tg_size,
                                        buf_input, (NSUInteger)N,
                                        buf_s1, buf_s2);
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
    }

    // Compute statistics
    double sum_t = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iters; ++i) {
        sum_t += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum_t / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%d", N);

    BenchResult r;
    r.kernel_name  = "reduction";
    r.problem_size = size_str;
    r.iterations   = iters;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;

    printCSVHeader();
    printCSVRow(r);

    // Effective bandwidth: 2 * N * sizeof(float) (read input + write result)
    double bandwidth_gb = 2.0 * (double)N * sizeof(float)
                          / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Effective bandwidth: %.2f GB/s  (avg %.4f ms)\n",
            bandwidth_gb, r.avg_ms);

    // -------------------------------------------------------------------
    // Verification — compare GPU result with CPU reference
    // -------------------------------------------------------------------
    float gpu_result = *((const float*)buf_result.contents);

    double cpu_result = 0.0;
    for (int i = 0; i < N; ++i) {
        cpu_result += (double)h_input[i];
    }

    double rel_err = fabs((double)gpu_result - cpu_result)
                     / (fabs(cpu_result) + 1e-10);
    fprintf(stderr, "GPU sum: %f, CPU sum: %f, relative error: %e\n",
            gpu_result, cpu_result, rel_err);

    if (rel_err < 1e-3) {
        fprintf(stderr, "PASS\n");
    } else {
        fprintf(stderr, "FAIL: relative error too large\n");
    }

    return (rel_err >= 1e-3) ? EXIT_FAILURE : EXIT_SUCCESS;

    } // @autoreleasepool
}
