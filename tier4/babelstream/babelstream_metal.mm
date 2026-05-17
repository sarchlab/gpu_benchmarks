#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

static int WARMUP = 0;

static const char *shaderSrcC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void copy_kernel(device const float* a [[buffer(0)]],\n"
    "                        device float* c [[buffer(1)]],\n"
    "                        uint i [[thread_position_in_grid]]) {\n"
    "    c[i] = a[i];\n"
    "}\n"
    "\n"
    "kernel void scale_kernel(device const float* c [[buffer(0)]],\n"
    "                         device float* b [[buffer(1)]],\n"
    "                         constant float& s [[buffer(2)]],\n"
    "                         uint i [[thread_position_in_grid]]) {\n"
    "    b[i] = s * c[i];\n"
    "}\n"
    "\n"
    "kernel void add_kernel(device const float* a [[buffer(0)]],\n"
    "                       device const float* b [[buffer(1)]],\n"
    "                       device float* c [[buffer(2)]],\n"
    "                       uint i [[thread_position_in_grid]]) {\n"
    "    c[i] = a[i] + b[i];\n"
    "}\n"
    "\n"
    "kernel void triad_kernel(device const float* b [[buffer(0)]],\n"
    "                         device const float* c [[buffer(1)]],\n"
    "                         device float* a [[buffer(2)]],\n"
    "                         constant float& s [[buffer(3)]],\n"
    "                         uint i [[thread_position_in_grid]]) {\n"
    "    a[i] = b[i] + s * c[i];\n"
    "}\n"
    "\n"
    "kernel void dot_kernel(device const float* a [[buffer(0)]],\n"
    "                       device const float* b [[buffer(1)]],\n"
    "                       device float* out [[buffer(2)]],\n"
    "                       uint i [[thread_position_in_grid]]) {\n"
    "    out[i] = a[i] * b[i];\n"
    "}\n";

static double mach_time_to_ms(uint64_t t) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)t * info.numer / info.denom / 1e6;
}

int main() {
    @autoreleasepool {

    setvbuf(stdout, NULL, _IONBF, 0);

    int n = 32 * 1024 * 1024;
    int threads_per_block = 256;
    float scalar = 2.0f;
    int num_times = 100;

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) n = atoi(env_val);
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);
    env_val = getenv("BENCH_PARAM_scalar");
    if (env_val) scalar = (float)atof(env_val);
    env_val = getenv("BENCH_PARAM_num_times");
    if (env_val) num_times = atoi(env_val);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "No Metal device\n"); return 1; }

    NSError *err = nil;
    NSString *shaderSrc = [NSString stringWithUTF8String:shaderSrcC];
    id<MTLLibrary> lib = [device newLibraryWithSource:shaderSrc options:nil error:&err];
    if (!lib) { fprintf(stderr, "Shader compile error: %s\n", [[err description] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Create pipeline states */
    id<MTLComputePipelineState> copy_ps  = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy_kernel"] error:nil];
    id<MTLComputePipelineState> scale_ps = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"scale_kernel"] error:nil];
    id<MTLComputePipelineState> add_ps   = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_kernel"] error:nil];
    id<MTLComputePipelineState> triad_ps = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"triad_kernel"] error:nil];
    id<MTLComputePipelineState> dot_ps   = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dot_kernel"] error:nil];

    size_t sz = (size_t)n * sizeof(float);
    id<MTLBuffer> bufa = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufb = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufc = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufd = [device newBufferWithLength:sz options:MTLResourceStorageModeShared]; /* dot output */

    id<MTLBuffer> s_buf = [device newBufferWithBytes:&scalar length:sizeof(float) options:MTLResourceStorageModeShared];

    /* Initialize */
    float *pa = (float*)bufa.contents;
    float *pb = (float*)bufb.contents;
    for (int i = 0; i < n; i++) { pa[i] = 1.0f; pb[i] = 2.0f; }

    MTLSize threadgroupSize = MTLSizeMake(threads_per_block, 1, 1);
    MTLSize gridSize = MTLSizeMake(n, 1, 1);

    double total_time_all = 0.0;

    /* Helper: run a kernel, warmup + time, emit JSON kernel event */
#define RUN_BENCH(kernel_name, ps, setup_block, bytes_expr) \
    do { \
        for (int _w = 0; _w < WARMUP; _w++) { \
            id<MTLCommandBuffer> cb = [queue commandBuffer]; \
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder]; \
            setup_block \
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize]; \
            [enc endEncoding]; \
            [cb commit]; [cb waitUntilCompleted]; \
        } \
        uint64_t t0 = mach_absolute_time(); \
        for (int _i = 0; _i < num_times; _i++) { \
            id<MTLCommandBuffer> cb = [queue commandBuffer]; \
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder]; \
            setup_block \
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize]; \
            [enc endEncoding]; \
            [cb commit]; [cb waitUntilCompleted]; \
        } \
        uint64_t t1 = mach_absolute_time(); \
        double ms = mach_time_to_ms(t1 - t0) / num_times; \
        double gbs = (bytes_expr) / (ms / 1000.0) / 1e9; \
        total_time_all += ms * num_times; \
        printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f," \
               "\"params\":{\"n\":%d,\"threads_per_block\":%d,\"scalar\":%.6f,\"num_times\":%d}}\n", \
               kernel_name, ms, n, threads_per_block, (double)scalar, num_times); \
        fprintf(stderr, "%s: %.3f ms, %.2f GB/s\n", kernel_name, ms, gbs); \
    } while(0)

    /* COPY */
    RUN_BENCH("copy_kernel", copy_ps,
        [enc setComputePipelineState:copy_ps];
        [enc setBuffer:bufa offset:0 atIndex:0];
        [enc setBuffer:bufc offset:0 atIndex:1];,
        2.0 * n * sizeof(float));

    /* SCALE */
    RUN_BENCH("scale_kernel", scale_ps,
        [enc setComputePipelineState:scale_ps];
        [enc setBuffer:bufc offset:0 atIndex:0];
        [enc setBuffer:bufb offset:0 atIndex:1];
        [enc setBuffer:s_buf offset:0 atIndex:2];,
        2.0 * n * sizeof(float));

    /* ADD */
    RUN_BENCH("add_kernel", add_ps,
        [enc setComputePipelineState:add_ps];
        [enc setBuffer:bufa offset:0 atIndex:0];
        [enc setBuffer:bufb offset:0 atIndex:1];
        [enc setBuffer:bufc offset:0 atIndex:2];,
        3.0 * n * sizeof(float));

    /* TRIAD */
    RUN_BENCH("triad_kernel", triad_ps,
        [enc setComputePipelineState:triad_ps];
        [enc setBuffer:bufb offset:0 atIndex:0];
        [enc setBuffer:bufc offset:0 atIndex:1];
        [enc setBuffer:bufa offset:0 atIndex:2];
        [enc setBuffer:s_buf offset:0 atIndex:3];,
        3.0 * n * sizeof(float));

    /* DOT */
    RUN_BENCH("dot_kernel", dot_ps,
        [enc setComputePipelineState:dot_ps];
        [enc setBuffer:bufa offset:0 atIndex:0];
        [enc setBuffer:bufb offset:0 atIndex:1];
        [enc setBuffer:bufd offset:0 atIndex:2];,
        2.0 * n * sizeof(float));

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[]}\n",
           total_time_all);

    return 0;

    } /* @autoreleasepool */
}
