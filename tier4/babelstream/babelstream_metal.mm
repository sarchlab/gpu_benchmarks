#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

#define N (32*1024*1024)
#define WARMUP 5
#define ITERS  5

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

    size_t sz = (size_t)N * sizeof(float);
    id<MTLBuffer> bufa = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufb = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufc = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufd = [device newBufferWithLength:sz options:MTLResourceStorageModeShared]; /* dot output */

    float s_val = 2.0f;
    id<MTLBuffer> s_buf = [device newBufferWithBytes:&s_val length:sizeof(float) options:MTLResourceStorageModeShared];

    /* Initialize */
    float *pa = (float*)bufa.contents;
    float *pb = (float*)bufb.contents;
    for (int i = 0; i < N; i++) { pa[i] = 1.0f; pb[i] = 2.0f; }

    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(N, 1, 1);

    /* Macro to run a kernel and time it */
#define RUN_BENCH(name, setup_block, bytes_expr) \
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
        for (int _i = 0; _i < ITERS; _i++) { \
            id<MTLCommandBuffer> cb = [queue commandBuffer]; \
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder]; \
            setup_block \
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize]; \
            [enc endEncoding]; \
            [cb commit]; [cb waitUntilCompleted]; \
        } \
        uint64_t t1 = mach_absolute_time(); \
        double ms = mach_time_to_ms(t1 - t0) / ITERS; \
        double gbs = (bytes_expr) / (ms / 1000.0) / 1e9; \
        printf("%s,%d,%.3f,%.2f\n", name, N, ms, gbs); \
    } while(0)

    /* COPY */
    RUN_BENCH("babelstream_copy",
        [enc setComputePipelineState:copy_ps];
        [enc setBuffer:bufa offset:0 atIndex:0];
        [enc setBuffer:bufc offset:0 atIndex:1];,
        2.0 * N * sizeof(float));

    /* SCALE */
    RUN_BENCH("babelstream_scale",
        [enc setComputePipelineState:scale_ps];
        [enc setBuffer:bufc offset:0 atIndex:0];
        [enc setBuffer:bufb offset:0 atIndex:1];
        [enc setBuffer:s_buf offset:0 atIndex:2];,
        2.0 * N * sizeof(float));

    /* ADD */
    RUN_BENCH("babelstream_add",
        [enc setComputePipelineState:add_ps];
        [enc setBuffer:bufa offset:0 atIndex:0];
        [enc setBuffer:bufb offset:0 atIndex:1];
        [enc setBuffer:bufc offset:0 atIndex:2];,
        3.0 * N * sizeof(float));

    /* TRIAD */
    RUN_BENCH("babelstream_triad",
        [enc setComputePipelineState:triad_ps];
        [enc setBuffer:bufb offset:0 atIndex:0];
        [enc setBuffer:bufc offset:0 atIndex:1];
        [enc setBuffer:bufa offset:0 atIndex:2];
        [enc setBuffer:s_buf offset:0 atIndex:3];,
        3.0 * N * sizeof(float));

    /* DOT */
    RUN_BENCH("babelstream_dot",
        [enc setComputePipelineState:dot_ps];
        [enc setBuffer:bufa offset:0 atIndex:0];
        [enc setBuffer:bufb offset:0 atIndex:1];
        [enc setBuffer:bufd offset:0 atIndex:2];,
        2.0 * N * sizeof(float));

    return 0;
}
