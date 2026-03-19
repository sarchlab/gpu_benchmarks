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
 * Timing is performed with mach_absolute_time for wall-clock accuracy on
 * Apple Silicon.  Results are reported in nanoseconds per access.
 *
 * Usage:
 *   ./cache_latency [array_bytes]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_array_bytes
 *   BENCH_PARAM_num_accesses
 *   BENCH_PARAM_num_iters
 *   BENCH_PARAM_rng_seed
 *
 * Build:
 *   make PLATFORM=metal
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
// Constants (defaults)
// ---------------------------------------------------------------------------

static const size_t   DEFAULT_ARRAY_BYTES  = 2ULL * 1024 * 1024;  // 2 MB
static const uint32_t DEFAULT_NUM_ACCESSES = 2000000u;
static const int      DEFAULT_NUM_ITERS    = 5;
static const uint32_t DEFAULT_RNG_SEED     = 42u;

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
 */
static void build_chain(uint32_t *arr, uint32_t N, uint32_t &start_idx,
                        uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (uint32_t i = 0; i < N; ++i) {
        arr[perm[i]] = perm[(i + 1u) % N];
    }
    start_idx = perm[0];
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
    long long array_bytes_ll = (long long)DEFAULT_ARRAY_BYTES;
    env_val = getenv("BENCH_PARAM_array_bytes");
    if (env_val) {
        array_bytes_ll = atoll(env_val);
    } else if (argc >= 2) {
        array_bytes_ll = atoll(argv[1]);
    }
    if (array_bytes_ll <= 0) {
        fprintf(stderr, "Invalid array_bytes\n");
        return EXIT_FAILURE;
    }
    size_t array_bytes = (size_t)array_bytes_ll;

    uint32_t num_accesses = DEFAULT_NUM_ACCESSES;
    env_val = getenv("BENCH_PARAM_num_accesses");
    if (env_val) num_accesses = (uint32_t)atoi(env_val);

    int num_iters = DEFAULT_NUM_ITERS;
    env_val = getenv("BENCH_PARAM_num_iters");
    if (env_val) num_iters = atoi(env_val);

    uint32_t rng_seed = DEFAULT_RNG_SEED;
    env_val = getenv("BENCH_PARAM_rng_seed");
    if (env_val) rng_seed = (uint32_t)atoi(env_val);

    const size_t   N     = array_bytes / sizeof(uint32_t);
    const uint32_t N_u32 = static_cast<uint32_t>(N);

    /* Select the default Metal device (GPU). */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }

    fprintf(stderr, "=== cache_latency microbenchmark ===\n");
    fprintf(stderr, "Device       : %s\n", device.name.UTF8String);
    fprintf(stderr, "Array size   : %zu bytes (%zu KB)\n",
            array_bytes, array_bytes / 1024);
    fprintf(stderr, "Num accesses : %u\n", num_accesses);
    fprintf(stderr, "Num iters    : %d\n", num_iters);
    fprintf(stderr, "RNG seed     : %u\n", rng_seed);
    fprintf(stderr, "Method       : single-thread pointer chasing\n\n");

    /* Load and compile the Metal shader. */
    NSError *err = nil;

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

    /* Build chain on host. */
    std::vector<uint32_t> h_arr(N);
    uint32_t start_idx = 0;
    build_chain(h_arr.data(), N_u32, start_idx, rng_seed);

    /* Allocate Metal buffers. */
    id<MTLBuffer> arr_buf =
        [device newBufferWithBytes:h_arr.data()
                           length:array_bytes
                          options:MTLResourceStorageModeShared];

    uint32_t params[2] = { start_idx, num_accesses };
    id<MTLBuffer> params_buf =
        [device newBufferWithBytes:params
                           length:sizeof(params)
                          options:MTLResourceStorageModeShared];

    id<MTLBuffer> result_buf =
        [device newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];

    auto dispatch_kernel = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:arr_buf    offset:0 atIndex:0];
        [enc setBuffer:params_buf offset:0 atIndex:1];
        [enc setBuffer:result_buf offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* Warm-up. */
    dispatch_kernel();

    /* Timed runs. */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < num_iters; ++iter) {
        dispatch_kernel();
    }
    uint64_t t1 = mach_absolute_time();

    double total_ns       = ticks_to_ns(t1 - t0);
    double total_accesses = (double)num_accesses * num_iters;
    double latency_ns     = total_ns / total_accesses;
    double total_ms       = total_ns / 1e6;
    double avg_ms         = total_ms / num_iters;

    fprintf(stderr, "Array %zu KB: latency = %.2f ns\n",
            array_bytes / 1024, latency_ns);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"pointer_chase_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"array_bytes\":%zu,\"num_accesses\":%u,"
           "\"num_iters\":%d,\"rng_seed\":%u}}\n",
           avg_ms, array_bytes, num_accesses, num_iters, rng_seed);

    /* JSON-lines output to stdout: summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"latency_ns\",\"value\":%.2f}]}\n",
           total_ms, latency_ns);

    fprintf(stderr, "\nDone.\n");

    } // @autoreleasepool
    return EXIT_SUCCESS;
}
