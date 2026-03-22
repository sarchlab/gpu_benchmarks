/**
 * altis_gups_metal.mm — Apple Metal host for GUPS benchmark.
 *
 * Measures random memory access throughput by performing random read-modify-write
 * operations on a large table of 64-bit entries. Each thread uses an xorshift
 * PRNG to generate random indices and XOR-updates the table.
 *
 * Output (stdout): JSON-lines protocol
 * Output (stderr): human-readable results
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
#include <vector>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double ticks_to_ms(uint64_t ticks) {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-6;
}

// ---------------------------------------------------------------------------
// CPU reference: perform random XOR updates
// ---------------------------------------------------------------------------

static void gups_cpu_ref(unsigned long long* table, int table_size,
                         int total_threads, int num_updates,
                         unsigned long long seed_base) {
    for (int tid = 0; tid < total_threads; ++tid) {
        unsigned long long state = seed_base + (unsigned long long)tid * 6364136223846793005ULL + 1;
        if (state == 0) state = 1;

        for (int u = 0; u < num_updates; ++u) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            int idx = (int)(state % (unsigned long long)table_size);
            table[idx] ^= state;
        }
    }
}

// ---------------------------------------------------------------------------
// Metal params struct (must match shader)
// ---------------------------------------------------------------------------

struct GUPSParams {
    uint32_t  table_size;
    uint32_t  num_updates;
    uint64_t  seed_base;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int table_size   = 1048576;
    int block_size   = 256;
    int num_updates  = 128;
    int num_threads  = 65536;
    int iterations   = 5;

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_table_size");
    if (env_val) table_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_updates");
    if (env_val) num_updates = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_threads");
    if (env_val) num_threads = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iterations");
    if (env_val) iterations = atoi(env_val);

    int iters      = iterations;
    int gridSize   = (num_threads + block_size - 1) / block_size;
    int actual_threads = gridSize * block_size;
    long long total_updates = (long long)actual_threads * num_updates;
    unsigned long long seed_base = 42ULL;

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "GUPS benchmark  |  Table size: %d (%.2f MB)  |  "
            "Updates/thread: %d  |  Threads: %d  |  Total updates: %lld  |  "
            "Iterations: 5 warmup + %d timed\n\n",
            table_size, (double)table_size * 8.0 / 1e6,
            num_updates, actual_threads, total_updates, iters);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader
    // -------------------------------------------------------------------
    NSError* err = nil;
    id<MTLLibrary> library = nil;

    NSString* exePath = [[NSBundle mainBundle] executablePath];
    NSString* dir     = [exePath stringByDeletingLastPathComponent];
    NSString* srcPath = [dir stringByAppendingPathComponent:@"altis_gups.metal"];

    if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
        NSString* metalSrc = [NSString stringWithContentsOfFile:srcPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        library = [device newLibraryWithSource:metalSrc options:opts error:&err];
        if (!library) {
            fprintf(stderr, "Metal compile error: %s\n",
                    err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Error: altis_gups.metal not found at %s\n",
                srcPath.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fnGUPS = [library newFunctionWithName:@"gups_kernel"];
    if (!fnGUPS) {
        fprintf(stderr, "Error: kernel function not found\n");
        return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> psoGUPS =
        [device newComputePipelineStateWithFunction:fnGUPS error:&err];
    if (!psoGUPS) {
        fprintf(stderr, "Error creating pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------
    // Allocate buffers
    // -------------------------------------------------------------------
    size_t table_bytes = (size_t)table_size * sizeof(unsigned long long);

    id<MTLBuffer> bufTable = [device newBufferWithLength:table_bytes
                                                options:MTLResourceStorageModeShared];

    // Initialize table
    std::vector<unsigned long long> h_table(table_size);
    for (int i = 0; i < table_size; ++i) {
        h_table[i] = (unsigned long long)i;
    }

    NSUInteger tgSize = (NSUInteger)block_size;
    NSUInteger gridN  = (NSUInteger)actual_threads;

    GUPSParams params;
    params.table_size  = (uint32_t)table_size;
    params.num_updates = (uint32_t)num_updates;
    params.seed_base   = seed_base;

    // Lambda: reset table
    auto reset_table = [&]() {
        memcpy(bufTable.contents, h_table.data(), table_bytes);
    };

    // Lambda: run GUPS kernel
    auto run_gups = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:psoGUPS];
        [enc setBuffer:bufTable offset:0 atIndex:0];
        [enc setBytes:&params length:sizeof(params) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(gridN, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // -------------------------------------------------------------------
    // Warmup (5 iterations)
    // -------------------------------------------------------------------
    for (int w = 0; w < 5; ++w) {
        reset_table();
        run_gups();
    }

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        reset_table();

        uint64_t t0 = mach_absolute_time();
        run_gups();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
    }

    double sum = 0.0;
    for (int i = 0; i < iters; ++i) sum += times[i];
    double avg_ms = sum / iters;
    double total_ms = sum;

    double gups = (double)total_updates / (avg_ms * 1e-3) / 1e9;

    // JSON-lines kernel event
    printf("{\"type\":\"kernel\",\"name\":\"gups_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"table_size\":%d,\"block_size\":%d,\"num_updates\":%d,"
           "\"num_threads\":%d,\"iterations\":%d}}\n",
           avg_ms, table_size, block_size, num_updates, num_threads, iterations);

    // JSON-lines summary event
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"gups\",\"value\":%.6f}]}\n",
           total_ms, gups);

    fprintf(stderr, "Average time: %.4f ms\n", avg_ms);
    fprintf(stderr, "Throughput:   %.6f GUPS (Giga Updates Per Second)\n", gups);

    // -------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------
    {
        reset_table();
        run_gups();

        const unsigned long long* gpuTable = (const unsigned long long*)bufTable.contents;

        std::vector<unsigned long long> cpu_table(h_table.begin(), h_table.end());
        gups_cpu_ref(cpu_table.data(), table_size, actual_threads, num_updates, seed_base);

        int verify_n = (table_size < 1024) ? table_size : 1024;
        int errors = 0;
        for (int i = 0; i < verify_n; ++i) {
            if (gpuTable[i] != cpu_table[i]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at index %d: GPU=%llu CPU=%llu\n",
                            i, gpuTable[i], cpu_table[i]);
                }
                errors++;
            }
        }

        if (errors > 0)
            fprintf(stderr, "FAIL: %d mismatches in first %d entries\n", errors, verify_n);
        else
            fprintf(stderr, "PASS\n");
    }

    return EXIT_SUCCESS;

    } // @autoreleasepool
}
