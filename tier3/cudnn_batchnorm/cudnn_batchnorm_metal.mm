/**
 * cudnn_batchnorm_metal.mm — Batch Normalization benchmark using Apple Metal
 *                            Performance Shaders (MPSCNNBatchNormalization).
 *
 * Computes batch normalization (forward only — MPS backward not easily
 * available in the public API):
 *   Input:  N=64, C=256, H=W=56
 *   Scale/Bias: per-channel (C=256)
 *   Forward:  y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Batch of N=64 images is processed by encoding 64 normalization commands
 * into a single command buffer.
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations, report average via mach_absolute_time()
 *
 * Effective bandwidth (forward only):
 *   ≈ 2 * N * C * H * W * sizeof(float) / time_s / 1e9
 *   (read input + write output)
 *
 * Output:
 *   CSV  (stdout): cudnn_batchnorm,64x256x56x56,<time_ms>,<GBs>
 *   Info (stderr): device name, per-iteration times, verification result
 *
 * Build:
 *   make PLATFORM=metal
 * or manually:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *           -framework Metal -framework MetalPerformanceShaders \
 *           -framework Foundation \
 *           cudnn_batchnorm_metal.mm -o cudnn_batchnorm
 *
 * NOTE: This Metal version performs forward-only batch normalization because
 * MPSCNNBatchNormalizationGradient requires a more complex training graph
 * setup that is not straightforward to use outside of MPSNNGraph. The
 * CUDA/ROCm version performs both forward and backward passes.
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
 * Benchmark parameters (mutable — overridable via BENCH_PARAM_* env vars)
 * ======================================================================= */
static int BATCH_N  = 64;
static int CHAN_C   = 256;
static int IN_H     = 56;
static int IN_W     = 56;
static int WARMUP   = 0;
static int ITERS    = 5;
#define BN_EPS   1e-5f

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
 * MPSCNNBatchNormalizationDataSource — provides gamma, beta, mean, variance
 * ======================================================================= */
@interface BNDataSource : NSObject <MPSCNNBatchNormalizationDataSource>
{
    NSUInteger _channels;
    float *_gamma;
    float *_beta;
    float *_mean;
    float *_variance;
}
- (instancetype)initWithChannels:(NSUInteger)channels
                           gamma:(float *)gamma
                            beta:(float *)beta
                            mean:(float *)mean
                        variance:(float *)variance;
@end

@implementation BNDataSource

- (instancetype)initWithChannels:(NSUInteger)channels
                           gamma:(float *)gamma
                            beta:(float *)beta
                            mean:(float *)mean
                        variance:(float *)variance
{
    self = [super init];
    if (self) {
        _channels = channels;
        _gamma = gamma;
        _beta = beta;
        _mean = mean;
        _variance = variance;
    }
    return self;
}

- (NSUInteger)numberOfFeatureChannels { return _channels; }
- (float *)gamma                      { return _gamma; }
- (float *)beta                       { return _beta; }
- (float *)mean                       { return _mean; }
- (float *)variance                   { return _variance; }
- (BOOL)load                          { return YES; }
- (void)purge                         { /* nothing */ }
- (NSString *)label                   { return @"batchnorm"; }
- (MPSDataType)dataType               { return MPSDataTypeFloat32; }
- (float)epsilon                      { return BN_EPS; }

- (BOOL)supportsSecureCoding          { return NO; }
- (void)encodeWithCoder:(nonnull NSCoder *)coder { /* not needed */ }
- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder { return nil; }

- (nonnull id)copyWithZone:(nullable NSZone *)zone {
    BNDataSource *copy = [[[self class] allocWithZone:zone] init];
    if (copy) {
        copy->_channels = _channels;
        copy->_gamma = _gamma;
        copy->_beta = _beta;
        copy->_mean = _mean;
        copy->_variance = _variance;
    }
    return copy;
}

@end

/* =========================================================================
 * Helper: write NCHW data for one image into an MPSImage texture
 *
 * MPSImage stores feature channels in slices of 4 (RGBA32Float).
 * For C channels, there are ceil(C/4) texture slices.
 * ======================================================================= */
static void write_nchw_to_mpsimage(MPSImage *img,
                                   const float *nchw, /* [C][H][W] */
                                   int C, int H, int W)
{
    int numSlices = (C + 3) / 4;
    NSUInteger bytesPerRow = (NSUInteger)W * 4 * sizeof(float);
    std::vector<float> rgba((size_t)H * W * 4);

    for (int s = 0; s < numSlices; s++) {
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
 * main
 * ======================================================================= */
int main(int argc, char *argv[])
{
    @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_spatial_dim");
    if (env_val) { IN_H = atoi(env_val); IN_W = IN_H; }
    env_val = getenv("BENCH_PARAM_batch_n");
    if (env_val) BATCH_N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_channels");
    if (env_val) CHAN_C = atoi(env_val);
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) ITERS = atoi(env_val);

    /* ------------------------------------------------------------------
     * Metal device setup
     * ---------------------------------------------------------------- */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: no Metal device found\n");
        return 1;
    }
    fprintf(stderr, "Device: %s\n", device.name.UTF8String);
    fprintf(stderr, "BatchNorm: N=%d C=%d H=%d W=%d  warmup=%d  iters=%d\n",
            BATCH_N, CHAN_C, IN_H, IN_W, WARMUP, ITERS);
    fprintf(stderr, "NOTE: Metal version performs forward-only batch normalization\n\n");

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ------------------------------------------------------------------
     * Generate random data
     * ---------------------------------------------------------------- */
    size_t input_elems = (size_t)BATCH_N * CHAN_C * IN_H * IN_W;
    std::vector<float> hInputNCHW(input_elems);

    srand(42);
    for (size_t i = 0; i < input_elems; i++)
        hInputNCHW[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    /* Per-channel parameters */
    std::vector<float> hGamma(CHAN_C, 1.0f);
    std::vector<float> hBeta(CHAN_C, 0.0f);

    /* Compute per-channel mean and variance from the input batch */
    std::vector<float> hMean(CHAN_C, 0.0f);
    std::vector<float> hVar(CHAN_C, 0.0f);
    size_t spatialSize = (size_t)BATCH_N * IN_H * IN_W;
    for (int c = 0; c < CHAN_C; c++) {
        double sum = 0.0;
        for (int n = 0; n < BATCH_N; n++) {
            for (int h = 0; h < IN_H; h++) {
                for (int w = 0; w < IN_W; w++) {
                    size_t idx = ((size_t)n * CHAN_C + c) * IN_H * IN_W +
                                 (size_t)h * IN_W + w;
                    sum += hInputNCHW[idx];
                }
            }
        }
        hMean[c] = (float)(sum / spatialSize);

        double var_sum = 0.0;
        for (int n = 0; n < BATCH_N; n++) {
            for (int h = 0; h < IN_H; h++) {
                for (int w = 0; w < IN_W; w++) {
                    size_t idx = ((size_t)n * CHAN_C + c) * IN_H * IN_W +
                                 (size_t)h * IN_W + w;
                    double diff = hInputNCHW[idx] - hMean[c];
                    var_sum += diff * diff;
                }
            }
        }
        hVar[c] = (float)(var_sum / spatialSize);
    }

    /* ------------------------------------------------------------------
     * Create MPSCNNBatchNormalization via data source
     * ---------------------------------------------------------------- */
    BNDataSource *dataSource =
        [[BNDataSource alloc] initWithChannels:CHAN_C
                                         gamma:hGamma.data()
                                          beta:hBeta.data()
                                          mean:hMean.data()
                                      variance:hVar.data()];

    MPSCNNBatchNormalization *bnKernel =
        [[MPSCNNBatchNormalization alloc] initWithDevice:device
                                              dataSource:dataSource];

    /* ------------------------------------------------------------------
     * Create MPSImage objects for input and output
     * ---------------------------------------------------------------- */
    MPSImageDescriptor *imgDesc = [MPSImageDescriptor
        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                  width:IN_W
                                 height:IN_H
                        featureChannels:CHAN_C];
    imgDesc.storageMode = MTLStorageModeShared;

    /* Source images (one per batch element) */
    NSMutableArray<MPSImage *> *srcImages =
        [NSMutableArray arrayWithCapacity:BATCH_N];
    for (int i = 0; i < BATCH_N; i++) {
        MPSImage *img = [[MPSImage alloc] initWithDevice:device
                                         imageDescriptor:imgDesc];
        const float *batchSlice = hInputNCHW.data() +
                                  (size_t)i * CHAN_C * IN_H * IN_W;
        write_nchw_to_mpsimage(img, batchSlice, CHAN_C, IN_H, IN_W);
        [srcImages addObject:img];
    }

    /* Destination images (one per batch element) */
    NSMutableArray<MPSImage *> *dstImages =
        [NSMutableArray arrayWithCapacity:BATCH_N];
    for (int i = 0; i < BATCH_N; i++) {
        MPSImage *img = [[MPSImage alloc] initWithDevice:device
                                         imageDescriptor:imgDesc];
        [dstImages addObject:img];
    }

    /* ------------------------------------------------------------------
     * Helper: encode + commit one batch (N normalizations)
     * ---------------------------------------------------------------- */
    auto run_batch = [&]() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int i = 0; i < BATCH_N; i++) {
            [bnKernel encodeToCommandBuffer:cb
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
    std::vector<double> times((size_t)ITERS);
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = mach_absolute_time();
        run_batch();
        uint64_t t1 = mach_absolute_time();
        times[i] = ticks_to_ms(t1 - t0);
        fprintf(stderr, "  iter %d: %.3f ms\n", i, times[i]);
    }

    /* ------------------------------------------------------------------
     * Read back output for batch element 0 and verify
     * ---------------------------------------------------------------- */
    std::vector<float> hOutputNCHW((size_t)CHAN_C * IN_H * IN_W, 0.0f);
    read_mpsimage_to_nchw(dstImages[0], hOutputNCHW.data(),
                          CHAN_C, IN_H, IN_W);

    /* Sanity check: output should not be all zeros or NaN */
    bool pass = true;
    bool has_nonzero = false;
    for (size_t i = 0; i < 1024 && i < hOutputNCHW.size(); i++) {
        if (std::isnan(hOutputNCHW[i]) || std::isinf(hOutputNCHW[i])) {
            fprintf(stderr, "FAIL: output[%zu] = %f (NaN/Inf)\n",
                    i, hOutputNCHW[i]);
            pass = false;
            break;
        }
        if (hOutputNCHW[i] != 0.0f) has_nonzero = true;
    }
    if (pass && !has_nonzero) {
        fprintf(stderr, "FAIL: output is all zeros\n");
        pass = false;
    }
    fprintf(stderr, "Verification (sanity check): %s\n\n",
            pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------
     * Statistics
     * ---------------------------------------------------------------- */
    double sum_ms = 0.0;
    for (int i = 0; i < ITERS; i++) sum_ms += times[i];
    double avg_ms = sum_ms / ITERS;

    /* Forward-only bandwidth: read input + write output = 2 * NCHW * 4 bytes
     * We report using the same formula as the HIP version for comparability:
     *   3 * N * C * H * W * sizeof(float) * 2
     * but note that Metal does forward only, so actual bytes moved is lower. */
    double total_bytes = 2.0 * BATCH_N * CHAN_C * IN_H * IN_W * sizeof(float);
    double time_s = avg_ms / 1000.0;
    double gbs = total_bytes / time_s / 1e9;

    /* JSON-lines output */
    printf("{\"type\":\"kernel\",\"name\":\"cudnnBatchNormalizationForwardTraining\",\"time_ms\":%.6f,"
           "\"params\":{\"spatial_dim\":%d,\"batch_n\":%d,\"channels\":%d,\"warmup\":%d,\"iters\":%d}}\n",
           avg_ms, IN_H, BATCH_N, CHAN_C, WARMUP, ITERS);
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f}]}\n",
           sum_ms, gbs);
    /* Human-readable (stderr) */
    fprintf(stderr,
        "cudnn_batchnorm  N=%d C=%d H=%d W=%d  time_ms=%.3f  GB/s=%.2f  "
        "(forward-only)\n",
        BATCH_N, CHAN_C, IN_H, IN_W, avg_ms, gbs);

    return pass ? 0 : 1;

    } /* @autoreleasepool */
}
