/**
 * cache_latency_metal.mm
 *
 * Apple Metal port of the cache_latency microbenchmark.
 *
 * Measures GPU cache/memory latency using pointer-chasing:
 *   A single GPU thread walks a random Hamiltonian cycle through an index
 *   array.  Because each load's address depends on the previous result, no
 *   memory-level parallelism is possible, and the total runtime directly
 *   reflects the round-trip latency of the targeted cache level.
 *
 * Three tests vary the array size:
 *   - L1 cache  :  16 KB  — fits in the L1 data cache
 *   - L2 cache  :   2 MB  — exceeds L1, fits in L2
 *   - Global mem:  64 MB  — exceeds L2; every access goes to DRAM
 *
 * Timing is performed with mach_absolute_time for wall-clock accuracy on
 * Apple Silicon.  Results are reported in nanoseconds per access.
 *
 * Build:
 *   make PLATFORM=metal
 *
 * Usage:
 *   ./cache_latency
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const size_t L1_ARRAY_BYTES   = 16ULL  * 1024;           //  16 KB
static const size_t L2_ARRAY_BYTES   = 2ULL   * 1024 * 1024;    //   2 MB
static const size_t GLOB_ARRAY_BYTES = 64ULL  * 1024 * 1024;    //  64 MB

static const uint32_t L1_ACCESSES   = 10000000u;  // 10 M steps
static const uint32_t L2_ACCESSES   =  2000000u;  //  2 M steps
static const uint32_t GLOB_ACCESSES =   300000u;  // 300 K steps

static const int NUM_ITERS = 5;
static const uint32_t RNG_SEED = 42u;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/** Convert mach_absolute_time ticks to nanoseconds. */
static double ticks_to_ns(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)ticks * (double)tb.numer / (double)tb.denom;
}

// ---------------------------------------------------------------------------
// Chain builder
// ---------------------------------------------------------------------------

/**
 * build_chain — build a random Hamiltonian pointer-chasing cycle.
 *
 * Constructs a single cycle of length N that visits every element exactly
 * once before returning to the start.  Random order defeats prefetchers.
 *
 * @param arr       Output host array of N uint32_t indices.
 * @param N         Number of elements.
 * @param start_idx (Out) The first index in the cycle.
 */
static void build_chain(uint32_t *arr, uint32_t N, uint32_t &start_idx)
{
    std::mt19937 rng(RNG_SEED);
    std::vector<uint32_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Wire the cycle: perm[i] → perm[(i+1) % N]
    for (uint32_t i = 0; i < N; ++i) {
        arr[perm[i]] = perm[(i + 1u) % N];
    }
    start_idx = perm[0];
}

// ---------------------------------------------------------------------------
// Per-test runner
// ---------------------------------------------------------------------------

struct TestConfig {
    const char *name;
    size_t      array_bytes;
    uint32_t    num_accesses;
    int         num_iters;
};

/**
 * run_latency_test — allocate buffers, warm up, time, and clean up one test.
 *
 * @param device  MTLDevice to use.
 * @param pso     Pre-built pipeline state for the pointer_chase kernel.
 * @param queue   Command queue.
 * @param cfg     Test configuration.
 * @return        Average latency in nanoseconds per access.
 */
static double run_latency_test(id<MTLDevice>               device,
                               id<MTLComputePipelineState> pso,
                               id<MTLCommandQueue>         queue,
                               const TestConfig           &cfg)
{
    const size_t   N     = cfg.array_bytes / sizeof(uint32_t);
    const uint32_t N_u32 = static_cast<uint32_t>(N);

    printf("  [%s] array = %zu KB, accesses = %u, iters = %d\n",
           cfg.name, cfg.array_bytes / 1024, cfg.num_accesses, cfg.num_iters);
    fflush(stdout);

    // --- Build the chain on the host ---
    std::vector<uint32_t> h_arr(N);
    uint32_t start_idx = 0;
    build_chain(h_arr.data(), N_u32, start_idx);

    // --- Allocate Metal buffers ---
    // Index array: shared so we can initialise it from the CPU.
    id<MTLBuffer> arr_buf =
        [device newBufferWithBytes:h_arr.data()
                           length:cfg.array_bytes
                          options:MTLResourceStorageModeShared];

    // params buffer: uint2 = { start_idx, num_accesses }
    uint32_t params[2] = { start_idx, cfg.num_accesses };
    id<MTLBuffer> params_buf =
        [device newBufferWithBytes:params
                           length:sizeof(params)
                          options:MTLResourceStorageModeShared];

    // result buffer: single uint32_t (prevents dead-code elimination)
    id<MTLBuffer> result_buf =
        [device newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];

    // Helper lambda: dispatch pointer_chase with 1 thread
    auto dispatch_kernel = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:arr_buf    offset:0 atIndex:0];
        [enc setBuffer:params_buf offset:0 atIndex:1];
        [enc setBuffer:result_buf offset:0 atIndex:2];
        // Launch exactly 1 thread — latency measurement, not throughput.
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // --- Warm-up ---
    dispatch_kernel();

    // --- Timed runs ---
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        dispatch_kernel();
    }
    uint64_t t1 = mach_absolute_time();

    double total_ns       = ticks_to_ns(t1 - t0);
    double total_accesses = (double)cfg.num_accesses * cfg.num_iters;
    return total_ns / total_accesses;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void)
{
    @autoreleasepool {

    // Select the default Metal device (GPU).
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    printf("=== cache_latency microbenchmark ===\n");
    printf("Device : %s\n", device.name.UTF8String);
    printf("Method : single-thread pointer chasing (random permutation)\n");
    printf("Each test: one warm-up + %d timed kernel launch(es)\n\n", NUM_ITERS);

    // --- Load and compile the Metal shader ---
    NSError *err = nil;

    // Locate cache_latency.metal next to the executable.
    NSString *metalSrc = nil;
    {
        NSString *exePath = [[NSBundle mainBundle] executablePath];
        NSString *dir     = [exePath stringByDeletingLastPathComponent];
        NSString *srcPath = [dir stringByAppendingPathComponent:
                             @"cache_latency.metal"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
            metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:nil];
        }
    }

    if (!metalSrc) {
        fprintf(stderr,
                "Error: cache_latency.metal not found next to the binary.\n"
                "Make sure to run from the build directory.\n");
        return EXIT_FAILURE;
    }

    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:metalSrc
                                                  options:opts
                                                    error:&err];
    if (!library) {
        fprintf(stderr, "Metal compile error: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"pointer_chase"];
    if (!fn) {
        fprintf(stderr, "Error: kernel 'pointer_chase' not found in shader\n");
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

    // --- Run the three cache-level tests ---
    const TestConfig tests[] = {
        { "L1 cache  ",  L1_ARRAY_BYTES,   L1_ACCESSES,   NUM_ITERS },
        { "L2 cache  ",  L2_ARRAY_BYTES,   L2_ACCESSES,   NUM_ITERS },
        { "Global mem",  GLOB_ARRAY_BYTES, GLOB_ACCESSES, NUM_ITERS },
    };
    const int NUM_TESTS = static_cast<int>(sizeof(tests) / sizeof(tests[0]));

    printf("Running tests...\n");
    double results[3] = {};

    for (int t = 0; t < NUM_TESTS; ++t) {
        results[t] = run_latency_test(device, pso, queue, tests[t]);
    }

    // --- Print summary ---
    printf("\n%-14s  %10s  %10s\n",
           "Cache level", "Array size", "Latency");
    printf("%-14s  %10s  %10s\n",
           "──────────", "──────────", "──────────");
    for (int t = 0; t < NUM_TESTS; ++t) {
        const char *unit;
        double sz;
        if (tests[t].array_bytes >= 1024 * 1024) {
            sz   = (double)tests[t].array_bytes / (1024.0 * 1024.0);
            unit = "MB";
        } else {
            sz   = (double)tests[t].array_bytes / 1024.0;
            unit = "KB";
        }
        printf("%-14s  %7.0f %-2s  %8.2f ns\n",
               tests[t].name, sz, unit, results[t]);
    }

    printf("\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
