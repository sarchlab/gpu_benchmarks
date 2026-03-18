/**
 * shoc_scan_metal.mm — Apple Metal host for the SHOC Scan benchmark.
 *
 * Exclusive prefix scan (sum): output[i] = sum(input[0..i-1])
 * Implements a work-efficient (Blelloch) multi-pass scan for arbitrarily
 * large arrays.  Derived from the SHOC benchmark suite.
 *
 * The Metal compute kernels live in shoc_scan.metal and are compiled at
 * runtime from source (loaded from the same directory as the binary).
 *
 * Multi-pass scan strategy:
 *   Phase 1: scan_block_kernel — per-threadgroup Blelloch scan
 *            → scanned output blocks + block_sums[]
 *   Phase 2: (recursive) scan block_sums[] → scanned_block_sums[]
 *   Phase 3: add_block_sums_kernel — add scanned_block_sums back
 *
 * Usage:
 *   ./shoc_scan [--array_size N] [--iterations I]
 *
 *   --array_size N   Number of float elements (default: 1048576 = 1 M)
 *   --iterations I   Timed kernel launches   (default: 20)
 *
 * Output (stdout): CSV — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
 * Output (stderr): Effective bandwidth in GB/s, verification PASS/FAIL
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
// Metal scan dispatch helpers
// ---------------------------------------------------------------------------

/**
 * dispatch_scan_block — Run one pass of scan_block_kernel.
 *
 * Each threadgroup handles tg_size*2 elements.
 * block_sums receives one float per threadgroup (the block's total).
 *
 * Returns the number of threadgroups dispatched (= length of block_sums used).
 */
static NSUInteger dispatch_scan_block(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso_scan,
    NSUInteger                  tg_size,
    id<MTLBuffer>               buf_input,
    id<MTLBuffer>               buf_output,
    id<MTLBuffer>               buf_block_sums,
    NSUInteger                  N)
{
    NSUInteger elems_per_tg = tg_size * 2;
    NSUInteger num_groups   = (N + elems_per_tg - 1) / elems_per_tg;

    uint32_t n32 = (uint32_t)N;

    id<MTLCommandBuffer>       cb  = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso_scan];
    [enc setBuffer:buf_input      offset:0 atIndex:0];
    [enc setBuffer:buf_output     offset:0 atIndex:1];
    [enc setBuffer:buf_block_sums offset:0 atIndex:2];
    [enc setBytes:&n32 length:sizeof(n32)  atIndex:3];
    // threadgroup memory: 2 * tg_size floats
    [enc setThreadgroupMemoryLength:elems_per_tg * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size,     1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return num_groups;
}

/**
 * dispatch_add_block_sums — Run one pass of add_block_sums_kernel.
 *
 * Adds scanned_block_sums[i] to every element in block i of data[].
 */
static void dispatch_add_block_sums(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso_add,
    NSUInteger                  tg_size,
    id<MTLBuffer>               buf_data,
    id<MTLBuffer>               buf_scanned_block_sums,
    NSUInteger                  N,
    NSUInteger                  num_groups)
{
    uint32_t n32 = (uint32_t)N;

    id<MTLCommandBuffer>       cb  = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso_add];
    [enc setBuffer:buf_data                offset:0 atIndex:0];
    [enc setBuffer:buf_scanned_block_sums  offset:0 atIndex:1];
    [enc setBytes:&n32 length:sizeof(n32)            atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size,     1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Forward declaration for recursive call
static void scan_recursive_metal(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso_scan,
    id<MTLComputePipelineState> pso_add,
    id<MTLDevice>               device,
    NSUInteger                  tg_size,
    id<MTLBuffer>               buf_input,
    id<MTLBuffer>               buf_output,
    NSUInteger                  N);

/**
 * scan_recursive_metal — Recursively performs exclusive prefix scan.
 *
 * Phase 1: per-block Blelloch scan  (scan_block_kernel)
 * Phase 2: recursively scan the block totals
 * Phase 3: propagate block prefix sums (add_block_sums_kernel)
 *
 * For arrays that fit in a single threadgroup (num_groups == 1) only
 * Phase 1 is needed.
 */
static void scan_recursive_metal(
    id<MTLCommandQueue>         queue,
    id<MTLComputePipelineState> pso_scan,
    id<MTLComputePipelineState> pso_add,
    id<MTLDevice>               device,
    NSUInteger                  tg_size,
    id<MTLBuffer>               buf_input,
    id<MTLBuffer>               buf_output,
    NSUInteger                  N)
{
    NSUInteger elems_per_tg = tg_size * 2;
    NSUInteger num_groups   = (N + elems_per_tg - 1) / elems_per_tg;

    // Allocate block_sums buffer (one float per threadgroup)
    id<MTLBuffer> buf_block_sums =
        [device newBufferWithLength:num_groups * sizeof(float)
                            options:MTLResourceStorageModeShared];

    // Phase 1: per-block scan
    dispatch_scan_block(queue, pso_scan, tg_size,
                        buf_input, buf_output, buf_block_sums, N);

    if (num_groups > 1) {
        // Phase 2: scan the block sums recursively
        id<MTLBuffer> buf_scanned_block_sums =
            [device newBufferWithLength:num_groups * sizeof(float)
                                options:MTLResourceStorageModeShared];

        scan_recursive_metal(queue, pso_scan, pso_add, device, tg_size,
                             buf_block_sums, buf_scanned_block_sums, num_groups);

        // Phase 3: add scanned block sums back to the output
        dispatch_add_block_sums(queue, pso_add, tg_size,
                                buf_output, buf_scanned_block_sums, N, num_groups);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {

    int N     = parseIntParam(argc, argv, "--array_size", 1048576);
    int iters = parseIterations(argc, argv);

    // Pad N to a multiple of (tg_size * 2) after we know tg_size.
    // We do the padding after pipeline creation.

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from source file next to the binary
    // -------------------------------------------------------------------
    NSError        *err     = nil;
    id<MTLLibrary>  library = nil;

    NSString *exePath = [[NSBundle mainBundle] executablePath];
    NSString *dir     = [exePath stringByDeletingLastPathComponent];
    NSString *srcPath = [dir stringByAppendingPathComponent:@"shoc_scan.metal"];

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
        fprintf(stderr, "Error: shoc_scan.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    // Build compute pipelines for the two kernels
    id<MTLFunction> fn_scan = [library newFunctionWithName:@"scan_block_kernel"];
    if (!fn_scan) {
        fprintf(stderr, "Error: kernel 'scan_block_kernel' not found\n");
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso_scan =
        [device newComputePipelineStateWithFunction:fn_scan error:&err];
    if (!pso_scan) {
        fprintf(stderr, "Error creating scan pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn_add = [library newFunctionWithName:@"add_block_sums_kernel"];
    if (!fn_add) {
        fprintf(stderr, "Error: kernel 'add_block_sums_kernel' not found\n");
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso_add =
        [device newComputePipelineStateWithFunction:fn_add error:&err];
    if (!pso_add) {
        fprintf(stderr, "Error creating add pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // Choose threadgroup size (cap at 256; must be power-of-2 for Blelloch sweep)
    NSUInteger tg_size = pso_scan.maxTotalThreadsPerThreadgroup;
    if (tg_size > 256) tg_size = 256;

    // Round tg_size down to the largest power-of-2 <= tg_size
    NSUInteger tg_pow2 = 1;
    while (tg_pow2 * 2 <= tg_size) tg_pow2 <<= 1;
    tg_size = tg_pow2;

    // Pad N_padded to next multiple of (tg_size * 2)
    NSUInteger elems_per_tg = tg_size * 2;
    NSUInteger N_padded = ((NSUInteger)N + elems_per_tg - 1) / elems_per_tg
                          * elems_per_tg;

    size_t bytes        = N_padded * sizeof(float);
    size_t bytes_orig   = (size_t)N * sizeof(float);

    fprintf(stderr, "Array size: %d floats (%.1f MiB, padded to %lu)\n",
            N, (double)bytes_orig / (1024.0 * 1024.0), (unsigned long)N_padded);
    fprintf(stderr, "Threadgroup size: %lu  |  Iterations: %d\n\n",
            (unsigned long)tg_size, iters);

    // -------------------------------------------------------------------
    // Allocate GPU buffers
    //   buf_input  — shared: CPU fills once, GPU reads (preserved across iters)
    //   buf_output — shared: GPU writes scan result, CPU reads for verification
    // -------------------------------------------------------------------
    id<MTLBuffer> buf_input  = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_output = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];

    // Initialize input with the same pattern as the HIP version
    float* h_input = (float*)buf_input.contents;
    for (NSUInteger i = 0; i < N_padded; ++i) {
        h_input[i] = (i < (NSUInteger)N) ? ((float)(i % 7) + 1.0f) : 0.0f;
    }

    // -------------------------------------------------------------------
    // Warmup
    // -------------------------------------------------------------------
    scan_recursive_metal(queue, pso_scan, pso_add, device, tg_size,
                         buf_input, buf_output, N_padded);

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = mach_absolute_time();
        scan_recursive_metal(queue, pso_scan, pso_add, device, tg_size,
                             buf_input, buf_output, N_padded);
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
    r.kernel_name  = "scan";
    r.problem_size = size_str;
    r.iterations   = iters;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;

    printCSVHeader();
    printCSVRow(r);

    // Effective bandwidth: 1 read (input) + 1 write (output) = 2 × N × sizeof(float)
    double bandwidth_gb = 2.0 * (double)N * sizeof(float)
                          / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Effective bandwidth: %.2f GB/s  (avg %.4f ms)\n",
            bandwidth_gb, r.avg_ms);

    // -------------------------------------------------------------------
    // Verification: compare GPU output against CPU exclusive scan
    // -------------------------------------------------------------------
    // Run one final scan for a clean result
    scan_recursive_metal(queue, pso_scan, pso_add, device, tg_size,
                         buf_input, buf_output, N_padded);

    const float* h_output = (const float*)buf_output.contents;

    // CPU reference
    std::vector<float> h_ref((size_t)N);
    h_ref[0] = 0.0f;
    for (int i = 1; i < N; ++i) {
        h_ref[i] = h_ref[i - 1] + h_input[i - 1];
    }

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_output[i] - h_ref[i]);
        float tol  = 1e-3f * fabsf(h_ref[i]) + 1e-5f;
        if (diff > tol) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: got %.6f, expected %.6f\n",
                        i, h_output[i], h_ref[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        fprintf(stderr, "FAIL: %d errors out of %d\n", errors, N);
    } else {
        fprintf(stderr, "PASS\n");
    }

    return (errors > 0) ? EXIT_FAILURE : EXIT_SUCCESS;

    } // @autoreleasepool
}
