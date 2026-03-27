/**
 * cudnn_conv_metal.mm — Forward 2D convolution benchmark using Apple Metal
 *                       Performance Shaders (MPSCNNConvolution).
 *
 * Computes forward convolution:
 *   Input:  N=32, C=3, H=W=224
 *   Filter: K=64, C=3, R=S=7, stride=2, pad=3
 *   Output: N=32, K=64, H_out=W_out=112
 *
 * Uses MPSCNNConvolution with a custom MPSCNNConvolutionDataSource.
 * Batch of 32 images is processed by encoding 32 convolution commands
 * into a single command buffer.
 *
 * Warmup: configurable iterations (not timed)
 * Timed:  single iteration (repetition handled by CLI)
 *
 * Verification: compare a small patch of output against CPU reference
 * GFLOPS = 2 * N * K * H_out * W_out * C * R * S / time_s / 1e9
 *
 * Output:
 *   CSV  (stdout): cudnn_conv,32x3x224x224,<time_ms>,<GFLOPS>
 *   Info (stderr): device name, per-iteration times, verification result
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework MetalPerformanceShaders \
 *           -framework Foundation \
 *           cudnn_conv_metal.mm -o cudnn_conv
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <mach/mach_time.h>

/* =========================================================================
 * Benchmark parameters (runtime-configurable via BENCH_PARAM_* env vars)
 * ======================================================================= */
static int BATCH_N  = 32;
static int IN_C     = 3;
static int IN_H     = 224;
static int IN_W     = 224;
static int OUT_K    = 64;
static int FILT_R   = 7;
static int FILT_S   = 7;
static const int STRIDE_V = 2;
static const int PAD_V    = 3;
static int OUT_H    = 112;
static int OUT_W    = 112;
static int WARMUP = 0;
/* =========================================================================
 * Timing helpers
 * ======================================================================= */
static double ticks_to_ms(uint64_t ticks)
{
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-6;
}

/* =========================================================================
 * MPSCNNConvolutionDataSource — provides weights to MPSCNNConvolution
 *
 * Weight layout for MPS:
 *   [outputFeatureChannels][kernelHeight][kernelWidth][inputFeatureChannels]
 *   i.e. OHWI (or KHWC in our notation).
 * ======================================================================= */
@interface ConvWeightDataSource : NSObject <MPSCNNConvolutionDataSource>
{
    MPSCNNConvolutionDescriptor *_desc;
    float *_weights;
    NSUInteger _weightCount;
}
- (instancetype)initWithKernelWidth:(NSUInteger)kW
                       kernelHeight:(NSUInteger)kH
               inputFeatureChannels:(NSUInteger)ifc
              outputFeatureChannels:(NSUInteger)ofc
                         strideX:(NSUInteger)sx
                         strideY:(NSUInteger)sy
                         weights:(float *)w;
@end

@implementation ConvWeightDataSource

- (instancetype)initWithKernelWidth:(NSUInteger)kW
                       kernelHeight:(NSUInteger)kH
               inputFeatureChannels:(NSUInteger)ifc
              outputFeatureChannels:(NSUInteger)ofc
                         strideX:(NSUInteger)sx
                         strideY:(NSUInteger)sy
                         weights:(float *)w
{
    self = [super init];
    if (self) {
        _desc = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:kW
                                      kernelHeight:kH
                              inputFeatureChannels:ifc
                             outputFeatureChannels:ofc];
        _desc.strideInPixelsX = sx;
        _desc.strideInPixelsY = sy;
        _weights = w;
        _weightCount = ofc * kH * kW * ifc;
    }
    return self;
}

- (MPSDataType)dataType                           { return MPSDataTypeFloat32; }
- (MPSCNNConvolutionDescriptor *)descriptor       { return _desc; }
- (void *)weights                                 { return _weights; }
- (float *)biasTerms                              { return NULL; }
- (BOOL)load                                      { return YES; }
- (void)purge                                     { /* nothing */ }
- (NSString *)label                               { return @"conv7x7"; }

- (nonnull id)copyWithZone:(nullable NSZone *)zone {
    ConvWeightDataSource *copy = [[[self class] allocWithZone:zone] init];
    if (copy) {
        copy->_desc = _desc;
        copy->_weights = _weights;
        copy->_weightCount = _weightCount;
    }
    return copy;
}

@end

/* =========================================================================
 * CPU reference convolution (small patch only)
 *
 * Input layout:  NCHW — input[((n*C + c)*H + h)*W + w]
 * Filter layout: KCHW — filter[((k*C + c)*R + r)*S + s]
 * Output layout: NKHW — output[((n*K + k)*OH + oh)*OW + ow]
 * ======================================================================= */
static void cpu_conv2d(int vn, int vk, int voh, int vow,
                       const float * __restrict__ input,
                       const float * __restrict__ filter,
                       float       * __restrict__ output)
{
    /* Output uses compact layout: output[((n*vk+k)*voh+oh)*vow+ow] */
    for (int n = 0; n < vn; n++) {
        for (int k = 0; k < vk; k++) {
            for (int oh = 0; oh < voh; oh++) {
                for (int ow = 0; ow < vow; ow++) {
                    float sum = 0.0f;
                    for (int c = 0; c < IN_C; c++) {
                        for (int r = 0; r < FILT_R; r++) {
                            for (int s = 0; s < FILT_S; s++) {
                                int ih = oh * STRIDE_V - PAD_V + r;
                                int iw = ow * STRIDE_V - PAD_V + s;
                                if (ih >= 0 && ih < IN_H &&
                                    iw >= 0 && iw < IN_W) {
                                    size_t in_idx = ((size_t)n * IN_C + c) *
                                                    IN_H * IN_W +
                                                    (size_t)ih * IN_W + iw;
                                    size_t f_idx  = ((size_t)k * IN_C + c) *
                                                    FILT_R * FILT_S +
                                                    (size_t)r * FILT_S + s;
                                    sum += input[in_idx] * filter[f_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = ((size_t)n * vk + k) *
                                     voh * vow +
                                     (size_t)oh * vow + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

/* =========================================================================
 * Helper: write NCHW data for one image into an MPSImage texture
 *
 * MPSImage stores feature channels in slices of 4 (RGBA32Float).
 * For C input channels, there are ceil(C/4) texture slices.
 * Each pixel in a slice has 4 floats: [ch_4s+0, ch_4s+1, ch_4s+2, ch_4s+3].
 * ======================================================================= */
static void write_nchw_to_mpsimage(MPSImage *img,
                                   const float *nchw, /* [C][H][W] */
                                   int C, int H, int W)
{
    int numSlices = (C + 3) / 4;
    NSUInteger bytesPerRow = (NSUInteger)W * 4 * sizeof(float);
    std::vector<float> rgba((size_t)H * W * 4);

    for (int s = 0; s < numSlices; s++) {
        /* Pack channels 4*s .. 4*s+3 into RGBA */
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int ci = 0; ci < 4; ci++) {
                    int c = s * 4 + ci;
                    size_t rgba_idx = ((size_t)h * W + w) * 4 + ci;
                    if (c < C) {
                        rgba[rgba_idx] = nchw[(size_t)c * H * W +
                                              (size_t)h * W + w];
                    } else {
                        rgba[rgba_idx] = 0.0f;
                    }
                }
            }
        }

        MTLRegion region = MTLRegionMake2D(0, 0,
                                           (NSUInteger)W, (NSUInteger)H);
        [img.texture replaceRegion:region
                       mipmapLevel:0
                             slice:(NSUInteger)s
                         withBytes:rgba.data()
                       bytesPerRow:bytesPerRow
                     bytesPerImage:bytesPerRow * (NSUInteger)H];
    }
}

/* =========================================================================
 * Helper: read MPSImage texture into NCHW buffer for one image
 * ======================================================================= */
static void read_mpsimage_to_nchw(MPSImage *img,
                                  float *nchw, /* [C][H][W] */
                                  int C, int H, int W)
{
    int numSlices = (C + 3) / 4;
    NSUInteger bytesPerRow = (NSUInteger)W * 4 * sizeof(float);
    std::vector<float> rgba((size_t)H * W * 4);

    for (int s = 0; s < numSlices; s++) {
        MTLRegion region = MTLRegionMake2D(0, 0,
                                           (NSUInteger)W, (NSUInteger)H);
        [img.texture getBytes:rgba.data()
                  bytesPerRow:bytesPerRow
                bytesPerImage:bytesPerRow * (NSUInteger)H
                   fromRegion:region
                  mipmapLevel:0
                        slice:(NSUInteger)s];

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int ci = 0; ci < 4; ci++) {
                    int c = s * 4 + ci;
                    if (c < C) {
                        nchw[(size_t)c * H * W + (size_t)h * W + w] =
                            rgba[((size_t)h * W + w) * 4 + ci];
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * Helper: convert filter from KCHW to KHWC (MPS weight layout)
 * ======================================================================= */
static void kchw_to_khwc(const float *kchw, float *khwc,
                          int K, int C, int R, int S)
{
    for (int k = 0; k < K; k++)
        for (int r = 0; r < R; r++)
            for (int s = 0; s < S; s++)
                for (int c = 0; c < C; c++)
                    khwc[((size_t)k * R * S + r * S + s) * C + c] =
                        kchw[((size_t)k * C + c) * R * S + r * S + s];
}

/* =========================================================================
 * main
 * ======================================================================= */
int main(int argc, char *argv[])
{
    @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_input_hw");
    if (env_val) { IN_H = atoi(env_val); IN_W = IN_H; }
    env_val = getenv("BENCH_PARAM_batch_n");
    if (env_val) BATCH_N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_in_channels");
    if (env_val) IN_C = atoi(env_val);
    env_val = getenv("BENCH_PARAM_out_channels");
    if (env_val) OUT_K = atoi(env_val);
    env_val = getenv("BENCH_PARAM_filter_size");
    if (env_val) { FILT_R = atoi(env_val); FILT_S = FILT_R; }
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);

    /* Recompute output dimensions */
    OUT_H = (IN_H + 2 * PAD_V - FILT_R) / STRIDE_V + 1;
    OUT_W = (IN_W + 2 * PAD_V - FILT_S) / STRIDE_V + 1;

    /* ------------------------------------------------------------------
     * Metal device setup
     * ---------------------------------------------------------------- */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return 1;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "Conv: N=%d C=%d H=%d W=%d K=%d R=%d S=%d stride=%d pad=%d\n",
            BATCH_N, IN_C, IN_H, IN_W, OUT_K, FILT_R, FILT_S,
            STRIDE_V, PAD_V);
    fprintf(stderr, "Output: %dx%dx%dx%d  warmup=%d\n\n",
            BATCH_N, OUT_K, OUT_H, OUT_W, WARMUP);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------
     * Generate random data in NCHW / KCHW format
     * ---------------------------------------------------------------- */
    size_t input_elems  = (size_t)BATCH_N * IN_C * IN_H * IN_W;
    size_t filter_elems = (size_t)OUT_K * IN_C * FILT_R * FILT_S;

    std::vector<float> hInputNCHW(input_elems);
    std::vector<float> hFilterKCHW(filter_elems);

    srand(42);
    for (size_t i = 0; i < input_elems; i++)
        hInputNCHW[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (size_t i = 0; i < filter_elems; i++)
        hFilterKCHW[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    /* Convert filter to KHWC layout for MPS */
    std::vector<float> hFilterKHWC(filter_elems);
    kchw_to_khwc(hFilterKCHW.data(), hFilterKHWC.data(),
                 OUT_K, IN_C, FILT_R, FILT_S);

    /* ------------------------------------------------------------------
     * Create MPSCNNConvolution via data source
     * ---------------------------------------------------------------- */
    ConvWeightDataSource *dataSource =
        [[ConvWeightDataSource alloc]
            initWithKernelWidth:FILT_S
                   kernelHeight:FILT_R
           inputFeatureChannels:IN_C
          outputFeatureChannels:OUT_K
                     strideX:STRIDE_V
                     strideY:STRIDE_V
                     weights:hFilterKHWC.data()];

    MPSCNNConvolution *conv =
        [[MPSCNNConvolution alloc] initWithDevice:device
                                          weights:dataSource];

    /* Set padding: offset (0,0,0) + zero edge mode gives pad=3 effect
     * for kernel=7. Default offset is (kernelWidth/2, kernelHeight/2, 0)
     * = (3, 3, 0), which means "valid" (no padding). Setting to (0,0,0)
     * shifts the kernel so it starts 3 pixels before the image edge. */
    conv.offset = (MPSOffset){0, 0, 0};
    conv.edgeMode = MPSImageEdgeModeZero;

    /* ------------------------------------------------------------------
     * Create MPSImage objects for input and output
     * ---------------------------------------------------------------- */
    MPSImageDescriptor *srcDesc = [MPSImageDescriptor
        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                  width:IN_W
                                 height:IN_H
                        featureChannels:IN_C];
    srcDesc.storageMode = MTLStorageModeShared;

    MPSImageDescriptor *dstDesc = [MPSImageDescriptor
        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                  width:OUT_W
                                 height:OUT_H
                        featureChannels:OUT_K];
    dstDesc.storageMode = MTLStorageModeShared;

    /* Source images (one per batch element) */
    NSMutableArray<MPSImage *> *srcImages =
        [NSMutableArray arrayWithCapacity:BATCH_N];
    for (int i = 0; i < BATCH_N; i++) {
        MPSImage *img = [[MPSImage alloc] initWithDevice:device
                                         imageDescriptor:srcDesc];
        /* Upload NCHW data for this batch element */
        const float *batchSlice = hInputNCHW.data() +
                                  (size_t)i * IN_C * IN_H * IN_W;
        write_nchw_to_mpsimage(img, batchSlice, IN_C, IN_H, IN_W);
        [srcImages addObject:img];
    }

    /* Destination images (one per batch element) */
    NSMutableArray<MPSImage *> *dstImages =
        [NSMutableArray arrayWithCapacity:BATCH_N];
    for (int i = 0; i < BATCH_N; i++) {
        MPSImage *img = [[MPSImage alloc] initWithDevice:device
                                         imageDescriptor:dstDesc];
        [dstImages addObject:img];
    }

    /* ------------------------------------------------------------------
     * Helper: encode + commit one batch (N convolutions)
     * ---------------------------------------------------------------- */
    auto run_batch = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int i = 0; i < BATCH_N; i++) {
            [conv encodeToCommandBuffer:cb
                            sourceImage:srcImages[i]
                       destinationImage:dstImages[i]];
        }
        [cb commit];
        [cb waitUntilCompleted];
    };

    /* ------------------------------------------------------------------
     * Warmup
     * ---------------------------------------------------------------- */
    for (int i = 0; i < WARMUP; i++) run_batch();

    /* ------------------------------------------------------------------
     * Timed iterations
     * ---------------------------------------------------------------- */
        uint64_t t0 = mach_absolute_time();
    run_batch();
    uint64_t t1 = mach_absolute_time();
    double total_ms = ticks_to_ms(t1 - t0);

    /* ------------------------------------------------------------------
     * Read back output for batch element 0 and verify
     * ---------------------------------------------------------------- */
    std::vector<float> hOutputNCHW((size_t)OUT_K * OUT_H * OUT_W, 0.0f);
    read_mpsimage_to_nchw(dstImages[0], hOutputNCHW.data(),
                          OUT_K, OUT_H, OUT_W);

    /* CPU reference — batch=0, first 4 output channels, 4×4 spatial */
    const int VN = 1, VK = 4, VOH = 4, VOW = 4;
    std::vector<float> cpuOut((size_t)VN * VK * VOH * VOW, 0.0f);
    cpu_conv2d(VN, VK, VOH, VOW,
               hInputNCHW.data(), hFilterKCHW.data(), cpuOut.data());

    bool pass = true;
    for (int k = 0; k < VK && pass; k++) {
        for (int oh = 0; oh < VOH && pass; oh++) {
            for (int ow = 0; ow < VOW && pass; ow++) {
                /* cpuOut uses compact layout: [k][oh][ow] */
                size_t ref_idx = (size_t)k * VOH * VOW +
                                 (size_t)oh * VOW + ow;
                /* hOutputNCHW uses full layout: [k][OUT_H][OUT_W] */
                size_t gpu_idx = (size_t)k * OUT_H * OUT_W +
                                 (size_t)oh * OUT_W + ow;
                float ref = cpuOut[ref_idx];
                float got = hOutputNCHW[gpu_idx];
                float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
                if (err > 1e-2f) {
                    fprintf(stderr,
                        "MISMATCH at [k=%d,oh=%d,ow=%d]: "
                        "ref=%.6f got=%.6f rel=%.4e\n",
                        k, oh, ow, ref, got, err);
                    pass = false;
                }
            }
        }
    }
    fprintf(stderr, "Verification (%d×%d×%d patch): %s\n\n",
            VK, VOH, VOW, pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------
     * Statistics
     * ---------------------------------------------------------------- */
    double avg_ms = total_ms;

    double flops = 2.0 * BATCH_N * OUT_K * OUT_H * OUT_W *
                   IN_C * FILT_R * FILT_S;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines output */
    printf("{\"type\":\"kernel\",\"name\":\"cudnnConvolutionForward\",\"time_ms\":%.6f,"
           "\"params\":{\"input_hw\":%d,\"batch_n\":%d,\"in_channels\":%d,"
           "\"out_channels\":%d,\"filter_size\":%d}}\n",
           avg_ms, IN_H, BATCH_N, IN_C, OUT_K, FILT_R);
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           total_ms, gflops);

    /* Human-readable (stderr) */
    fprintf(stderr,
        "cudnn_conv  N=%d C=%d H=%d W=%d K=%d R=%d S=%d stride=%d pad=%d  "
        "time_ms=%.3f  GFLOPS=%.2f\n",
        BATCH_N, IN_C, IN_H, IN_W, OUT_K, FILT_R, FILT_S,
        STRIDE_V, PAD_V, avg_ms, gflops);

    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
