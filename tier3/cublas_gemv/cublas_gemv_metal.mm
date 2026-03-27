/**
 * cublas_gemv_metal.mm — FP32 GEMV benchmark using a custom Apple Metal kernel.
 *
 * Computes: y = A * x
 *   where A is M×N (row-major, FP32), x is N-vector, y is M-vector
 *   M = N = 4096
 *
 * Metal kernel: one thread per output row; each thread dot-products one row
 * of A against x and writes to y[row].
 *
 * The Metal shader is compiled at runtime from an embedded source string
 * (no separate .metallib file required — binary is self-contained).
 *
 * Warmup: 5 command buffers (not timed)
 * Timed:  5 command buffers, report average via mach_absolute_time()
 *
 * Verification: compare all M elements of y against CPU GEMV, max rel error < 1e-4
 * GFLOPS = 2 * M * N / time_s / 1e9
 * GB/s   = (M*N + M + N) * sizeof(float) / time_s / 1e9
 *
 * Output:
 *   CSV    (stdout): cublas_gemv,4096,<time_ms>,<GBs>,<GFLOPS>
 *   Info   (stderr): device name, per-iteration times, verification result
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework Foundation \
 *           cublas_gemv_metal.mm -o cublas_gemv
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

/* -------------------------------------------------------------------------
 * Benchmark default parameters (overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * Embedded Metal shader source
 * One thread per output row; computes dot product of row i of A with x.
 * ---------------------------------------------------------------------- */
static const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void sgemv_kernel(
    device const float* A   [[ buffer(0) ]],
    device const float* x   [[ buffer(1) ]],
    device       float* y   [[ buffer(2) ]],
    constant     uint&  N   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    uint base = gid * N;
    for (uint j = 0; j < N; j++)
        sum += A[base + j] * x[j];
    y[gid] = sum;
}
)METAL";

/* -------------------------------------------------------------------------
 * Timing helpers
 * ---------------------------------------------------------------------- */
static double ticks_to_ms(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-6;
}

/* -------------------------------------------------------------------------
 * CPU reference GEMV
 * ---------------------------------------------------------------------- */
static void cpu_sgemv(const float * __restrict__ A,
                      const float * __restrict__ x,
                      float       * __restrict__ y,
                      int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        for (int j = 0; j < cols; j++)
            s += A[(size_t)i * cols + j] * x[j];
        y[i] = s;
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    @autoreleasepool {

    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Benchmark parameters (overridable via BENCH_PARAM_* env vars)       */
    /* ------------------------------------------------------------------ */
    int M = 4096;
    int N = 4096;
    int WARMUP = 0;
    int ITERS = 5;
    float alpha_param = 1.0f;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_dim");
    if (env_val) { M = atoi(env_val); N = M; }
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) ITERS = atoi(env_val);
    env_val = getenv("BENCH_PARAM_alpha");
    if (env_val) alpha_param = (float)atof(env_val);

    /* ------------------------------------------------------------------ */
    /* Metal device setup                                                  */
    /* ------------------------------------------------------------------ */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "M=%d  N=%d  warmup=%d  iters=%d\n\n", M, N, WARMUP, ITERS);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------ */
    /* Compile Metal shader from embedded source string                    */
    /* ------------------------------------------------------------------ */
    NSError* err = nil;
    NSString* metalSrc = [NSString stringWithUTF8String:kMetalSource];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:metalSrc options:opts error:&err];
    if (!library) {
        fprintf(stderr, "Metal compile error: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"sgemv_kernel"];
    if (!fn) {
        fprintf(stderr, "Error: kernel 'sgemv_kernel' not found in library\n");
        return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "Error creating pipeline: %s\n",
                err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    /* ------------------------------------------------------------------ */
    /* Allocate Metal buffers (Shared storage for easy CPU init/readback)  */
    /* ------------------------------------------------------------------ */
    size_t bytesA = (size_t)M * N * sizeof(float);
    size_t bytesX = (size_t)N     * sizeof(float);
    size_t bytesY = (size_t)M     * sizeof(float);

    id<MTLBuffer> bufA = [device newBufferWithLength:bytesA
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufX = [device newBufferWithLength:bytesX
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufY = [device newBufferWithLength:bytesY
                                             options:MTLResourceStorageModeShared];

    if (!bufA || !bufX || !bufY) {
        fprintf(stderr, "Error: failed to allocate Metal buffers\n");
        return EXIT_FAILURE;
    }

    /* ------------------------------------------------------------------ */
    /* Initialize data                                                     */
    /* ------------------------------------------------------------------ */
    float* hA = (float*)bufA.contents;
    float* hx = (float*)bufX.contents;

    srand(42);
    for (int i = 0; i < M * N; i++) hA[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (int i = 0; i < N;     i++) hx[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    memset(bufY.contents, 0, bytesY);

    /* Constant: N as uint */
    uint32_t N_val = (uint32_t)N;

    /* Threadgroup size */
    NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
    if (tg_size > 64) tg_size = 64;

    /* Helper lambda: encode and commit one GEMV dispatch */
    auto dispatch_gemv = [&]() {
        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufY offset:0 atIndex:2];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)M, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* ------------------------------------------------------------------ */
    /* Warmup (not measured)                                               */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) dispatch_gemv();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    std::vector<double> times((size_t)ITERS);
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = mach_absolute_time();
        dispatch_gemv();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
        fprintf(stderr, "  iter %d: %.4f ms\n", i, times[i]);
    }

    /* ------------------------------------------------------------------ */
    /* Verify all M elements vs CPU GEMV                                  */
    /* ------------------------------------------------------------------ */
    std::vector<float> hy_ref((size_t)M, 0.0f);
    cpu_sgemv(hA, hx, hy_ref.data(), M, N);

    const float* hy_gpu = (const float*)bufY.contents;
    bool pass = true;
    for (int i = 0; i < M; i++) {
        float ref = hy_ref[i];
        float rel = fabsf(hy_gpu[i] - ref) / (fabsf(ref) + 1e-6f);
        if (rel > 1e-3f) {
            fprintf(stderr, "MISMATCH y[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, hy_gpu[i], ref, rel);
            pass = false;
            break;
        }
    }
    fprintf(stderr, "Verification (all %d elements): %s\n\n", M,
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------ */
    /* Statistics                                                          */
    /* ------------------------------------------------------------------ */
    double sum_ms = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < ITERS; i++) {
        sum_ms += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg_ms = sum_ms / ITERS;

    double variance = 0.0;
    for (int i = 0; i < ITERS; i++) {
        double d = times[i] - avg_ms;
        variance += d * d;
    }
    double stddev = (ITERS > 1) ? sqrt(variance / (ITERS - 1)) : 0.0;

    /* GFLOPS = 2 * M * N multiply-adds */
    double gflops = 2.0 * M * N / (avg_ms * 1e-3) / 1e9;

    /* GB/s = matrix A (M*N floats) + vector x (N floats) + vector y (M floats) */
    double bytes_bw = ((double)M * N + M + N) * sizeof(float);
    double gbps     = bytes_bw / (avg_ms * 1e-3) / 1e9;

    fprintf(stderr, "GB/s: %.2f  GFLOPS: %.2f  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gbps, gflops, avg_ms, mn, mx, stddev);

    /* JSON-lines kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"cublasSgemv\",\"time_ms\":%.6f,"
           "\"params\":{\"matrix_dim\":%d,\"warmup\":%d,\"iters\":%d,\"alpha\":%.6f}}\n",
           avg_ms, M, WARMUP, ITERS, alpha_param);
    /* JSON-lines summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f},{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           sum_ms, gbps, gflops);

    /* Human-readable */
    fprintf(stderr, "cublas_gemv  M=%d N=%d  time_ms=%.4f  GB/s=%.2f  GFLOPS=%.2f\n",
            M, N, avg_ms, gbps, gflops);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;

    } /* @autoreleasepool */
}
