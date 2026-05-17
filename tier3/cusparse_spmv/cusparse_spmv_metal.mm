/**
 * cusparse_spmv_metal.mm — Apple Metal host for the cuSPARSE SpMV benchmark.
 *
 * Computes: y = A * x
 *   where A is a synthetic CSR sparse matrix with exactly NNZ_PER_ROW
 *   non-zeros per row and NUM_ROWS rows.
 *
 * Metal has no equivalent to cuSPARSE, so this file uses a custom CSR SpMV
 * Metal compute shader (same kernel as tier2/shoc_spmv).  The benchmark
 * interface, parameters, and output format match the CUDA/ROCm version.
 *
 * Each GPU thread computes one row of the output vector.
 * The Metal shader is compiled at runtime from an embedded source string
 * (no separate .metallib file required — binary is self-contained).
 *
 * Parameters (from env vars BENCH_PARAM_*):
 *   num_rows    — number of rows (and columns) in the square sparse matrix
 *   nnz_per_row — non-zeros per row
 *   warmup      — warmup iterations
 *
 * Metrics:
 *   GB/s   = NNZ * 12.0 / time_s / 1e9
 *   GFLOPS = 2.0 * NNZ  / time_s / 1e9
 *
 * Output (stdout): JSON-lines protocol
 * Output (stderr): device info, per-iteration times, verification result
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
// Embedded Metal shader source
// (Custom CSR SpMV kernel — Metal has no cuSPARSE equivalent)
// ---------------------------------------------------------------------------

static const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr(
    device const int*   row_ptr [[ buffer(0) ]],
    device const int*   col_idx [[ buffer(1) ]],
    device const float* values  [[ buffer(2) ]],
    device const float* x       [[ buffer(3) ]],
    device       float* y       [[ buffer(4) ]],
    constant     uint&  num_rows[[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= num_rows) return;
    float sum = 0.0f;
    int start = row_ptr[gid];
    int end   = row_ptr[gid + 1];
    for (int k = start; k < end; ++k)
        sum += values[k] * x[col_idx[k]];
    y[gid] = sum;
}
)METAL";

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
// CPU SpMV reference for correctness verification
// ---------------------------------------------------------------------------

static void cpu_spmv(const int* row_ptr, const int* col_idx,
                     const float* values, const float* x,
                     float* y, int num_rows)
{
    for (int row = 0; row < num_rows; ++row) {
        float sum = 0.0f;
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
            sum += values[k] * x[col_idx[k]];
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);

    // -------------------------------------------------------------------
    // Default parameters (overridden by BENCH_PARAM_* env vars)
    // -------------------------------------------------------------------
    int num_rows    = 16384;
    int nnz_per_row = 32;
    int warmup      = 0;
    const char *env_val;
    env_val = getenv("BENCH_PARAM_num_rows");
    if (env_val) num_rows = atoi(env_val);
    env_val = getenv("BENCH_PARAM_nnz_per_row");
    if (env_val) nnz_per_row = atoi(env_val);
    const int num_cols = num_rows;
    const int nnz      = num_rows * nnz_per_row;

    // -------------------------------------------------------------------
    // Metal device setup
    // -------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Platform: Metal (custom CSR SpMV kernel — no cuSPARSE equivalent)\n");
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Rows: %d  |  NNZ_per_row: %d  |  Total NNZ: %d\n"
                    "Warmup: %d\n\n",
            num_rows, nnz_per_row, nnz, warmup);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // -------------------------------------------------------------------
    // Compile Metal shader from embedded source string
    // -------------------------------------------------------------------
    NSError* err = nil;
    NSString* metalSrc = [NSString stringWithUTF8String:kMetalSource];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:metalSrc options:opts error:&err];
    if (!library) {
        fprintf(stderr, "Metal compile error: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"spmv_csr"];
    if (!fn) {
        fprintf(stderr, "Error: kernel 'spmv_csr' not found in library\n");
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------
    // Allocate Metal buffers (Shared storage for easy CPU init/readback)
    // -------------------------------------------------------------------
    size_t row_ptr_bytes = (size_t)(num_rows + 1) * sizeof(int);
    size_t col_idx_bytes = (size_t)nnz            * sizeof(int);
    size_t val_bytes     = (size_t)nnz            * sizeof(float);
    size_t vec_bytes     = (size_t)num_rows        * sizeof(float);

    id<MTLBuffer> buf_row_ptr = [device newBufferWithLength:row_ptr_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_col_idx = [device newBufferWithLength:col_idx_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_values  = [device newBufferWithLength:val_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_x       = [device newBufferWithLength:vec_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_y       = [device newBufferWithLength:vec_bytes
                                                    options:MTLResourceStorageModeShared];

    // -------------------------------------------------------------------
    // Initialize CSR matrix and input vector
    // -------------------------------------------------------------------
    int*   h_row_ptr = (int*)  buf_row_ptr.contents;
    int*   h_col_idx = (int*)  buf_col_idx.contents;
    float* h_values  = (float*)buf_values.contents;
    float* h_x       = (float*)buf_x.contents;

    srand(42);
    for (int i = 0; i <= num_rows; ++i)
        h_row_ptr[i] = i * nnz_per_row;

    for (int k = 0; k < nnz; ++k) {
        h_col_idx[k] = rand() % num_cols;
        h_values[k]  = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < num_rows; ++i)
        h_x[i] = (float)rand() / (float)RAND_MAX;

    uint32_t num_rows_u = (uint32_t)num_rows;

    // Threadgroup size
    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > 256) tg_size = 256;

    // Lambda: encode and commit one SpMV dispatch
    auto dispatch_spmv = [&]() {
        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_row_ptr offset:0 atIndex:0];
        [enc setBuffer:buf_col_idx offset:0 atIndex:1];
        [enc setBuffer:buf_values  offset:0 atIndex:2];
        [enc setBuffer:buf_x       offset:0 atIndex:3];
        [enc setBuffer:buf_y       offset:0 atIndex:4];
        [enc setBytes:&num_rows_u length:sizeof(num_rows_u) atIndex:5];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)num_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // -------------------------------------------------------------------
    // Warmup (not measured)
    // -------------------------------------------------------------------
    for (int w = 0; w < warmup; w++)
        dispatch_spmv();

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    uint64_t t0 = mach_absolute_time();
    dispatch_spmv();
    uint64_t t1 = mach_absolute_time();
    double total_ms = ticks_to_ms(t1 - t0);

    // -------------------------------------------------------------------
    // Correctness check: compare GPU y with CPU SpMV (first 32 elements)
    // -------------------------------------------------------------------
    std::vector<float> h_y_ref((size_t)num_rows);
    cpu_spmv(h_row_ptr, h_col_idx, h_values, h_x, h_y_ref.data(), num_rows);

    const float* h_y_gpu = (const float*)buf_y.contents;
    bool pass = true;
    int check_n = 32;
    for (int i = 0; i < check_n; ++i) {
        float ref = h_y_ref[i];
        float rel = fabsf(h_y_gpu[i] - ref) / (fabsf(ref) + 1e-6f);
        if (rel > 1e-4f) {
            fprintf(stderr, "MISMATCH y[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, h_y_gpu[i], ref, rel);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (first %d elements): %s\n\n",
            check_n, pass ? "PASS" : "FAIL");

    // -------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------
    double avg_ms = total_ms;

    double time_s = avg_ms * 1e-3;
    double gbps   = (double)nnz * 12.0 / time_s / 1e9;
    double gflops = 2.0 * (double)nnz / time_s / 1e9;

    fprintf(stderr,
            "GB/s: %.2f  GFLOPS: %.2f  (%.4f ms)\n",
            gbps, gflops, avg_ms);

    // JSON-lines: kernel event
    printf("{\"type\":\"kernel\",\"name\":\"cusparseSpMV\",\"time_ms\":%.6f,"
           "\"params\":{\"num_rows\":%d,\"nnz_per_row\":%d,\"warmup\":%d}}\n",
           avg_ms, num_rows, nnz_per_row, warmup);

    // JSON-lines: summary event
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f},{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           total_ms, gbps, gflops);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;

    } // @autoreleasepool
}
