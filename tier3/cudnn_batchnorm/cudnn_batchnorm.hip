/**
 * cudnn_batchnorm.hip — Batch Normalization forward + backward benchmark
 *                       using cuDNN (NVIDIA) or MIOpen (AMD).
 *
 * Computes batch normalization (spatial mode):
 *   Input:  N=64, C=256, H=W=56  (NCHW layout, FP32)
 *   Scale/Bias: per-channel (C=256)
 *   Forward:  y = scale * (x - mean) / sqrt(var + eps) + bias
 *   Backward: compute grad_input, grad_scale, grad_bias
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations, report average
 *
 * Effective bandwidth = total_bytes_read_and_written / time_s / 1e9
 *   For fwd+bwd ≈ 3 * N * C * H * W * sizeof(float) * 2 / time_s / 1e9
 *
 * Output:
 *   cudnn_batchnorm,64x256x56x56,<time_ms>,<GBs>
 *
 * Build:
 *   CUDA: nvcc -O3 -std=c++14 -x cu -DUSE_CUDNN  cudnn_batchnorm.hip -lcudnn -o cudnn_batchnorm
 *   ROCm: hipcc -O3 -std=c++14     -DUSE_MIOPEN cudnn_batchnorm.hip -lMIOpen -o cudnn_batchnorm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* =========================================================================
 * Platform abstraction
 * ======================================================================= */

#ifdef USE_CUDNN
#  include <cudnn.h>
#  include <cuda_runtime.h>

#  define GPU_CHECK(x) do {                                                  \
       cudaError_t _e = (x);                                                 \
       if (_e != cudaSuccess) {                                               \
           fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#  define DNN_CHECK(x) do {                                                  \
       cudnnStatus_t _e = (x);                                               \
       if (_e != CUDNN_STATUS_SUCCESS) {                                      \
           fprintf(stderr, "cuDNN error %s:%d: %s\n",                        \
                   __FILE__, __LINE__, cudnnGetErrorString(_e));              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

typedef cudaEvent_t gpu_event_t;
#  define GPU_EVENT_CREATE(e)       GPU_CHECK(cudaEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)       GPU_CHECK(cudaEventRecord(e))
#  define GPU_EVENT_SYNC(e)         GPU_CHECK(cudaEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(cudaEventElapsedTime(&(ms),(a),(b)))
#  define GPU_DEVICE_SYNC()         GPU_CHECK(cudaDeviceSynchronize())
#  define GPU_MALLOC(p,sz)          GPU_CHECK(cudaMalloc((void**)&(p),(sz)))
#  define GPU_FREE(p)               cudaFree(p)
#  define GPU_MEMCPY_H2D(dst,src,sz) GPU_CHECK(cudaMemcpy((dst),(src),(sz),cudaMemcpyHostToDevice))
#  define GPU_MEMCPY_D2H(dst,src,sz) GPU_CHECK(cudaMemcpy((dst),(src),(sz),cudaMemcpyDeviceToHost))
#  define GPU_MEMSET(p,v,sz)        GPU_CHECK(cudaMemset((p),(v),(sz)))

#elif defined(USE_MIOPEN)
#  include <miopen/miopen.h>
#  include <hip/hip_runtime.h>

#  define GPU_CHECK(x) do {                                                  \
       hipError_t _e = (x);                                                  \
       if (_e != hipSuccess) {                                                \
           fprintf(stderr, "HIP error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, hipGetErrorString(_e));                \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#  define DNN_CHECK(x) do {                                                  \
       miopenStatus_t _e = (x);                                              \
       if (_e != miopenStatusSuccess) {                                       \
           fprintf(stderr, "MIOpen error %s:%d: %d\n",                       \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

typedef hipEvent_t gpu_event_t;
#  define GPU_EVENT_CREATE(e)       GPU_CHECK(hipEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)       GPU_CHECK(hipEventRecord((e), 0))
#  define GPU_EVENT_SYNC(e)         GPU_CHECK(hipEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a,b,ms) GPU_CHECK(hipEventElapsedTime(&(ms),(a),(b)))
#  define GPU_DEVICE_SYNC()         GPU_CHECK(hipDeviceSynchronize())
#  define GPU_MALLOC(p,sz)          GPU_CHECK(hipMalloc((void**)&(p),(sz)))
#  define GPU_FREE(p)               hipFree(p)
#  define GPU_MEMCPY_H2D(dst,src,sz) GPU_CHECK(hipMemcpy((dst),(src),(sz),hipMemcpyHostToDevice))
#  define GPU_MEMCPY_D2H(dst,src,sz) GPU_CHECK(hipMemcpy((dst),(src),(sz),hipMemcpyDeviceToHost))
#  define GPU_MEMSET(p,v,sz)        GPU_CHECK(hipMemset((p),(v),(sz)))

#else
#  error "Define USE_CUDNN or USE_MIOPEN"
#endif

/* =========================================================================
 * Benchmark parameters (mutable — overridable via BENCH_PARAM_* env vars)
 * ======================================================================= */
static int BATCH_N  = 64;
static int CHAN_C   = 256;
static int IN_H     = 56;
static int IN_W     = 56;
static int WARMUP   = 0;
static int ITERS    = 5;
#define BN_EPS   1e-5

/* =========================================================================
 * main
 * ======================================================================= */
int main(void)
{
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
     * Host allocation and initialisation
     * ---------------------------------------------------------------- */
    size_t tensor_elems = (size_t)BATCH_N * CHAN_C * IN_H * IN_W;
    size_t chan_elems    = (size_t)CHAN_C;

    float *hInput     = (float *)malloc(tensor_elems * sizeof(float));
    float *hOutput    = (float *)malloc(tensor_elems * sizeof(float));
    float *hGradOut   = (float *)malloc(tensor_elems * sizeof(float));
    float *hGradIn    = (float *)malloc(tensor_elems * sizeof(float));
    float *hScale     = (float *)malloc(chan_elems * sizeof(float));
    float *hBias      = (float *)malloc(chan_elems * sizeof(float));
    float *hRunMean   = (float *)malloc(chan_elems * sizeof(float));
    float *hRunVar    = (float *)malloc(chan_elems * sizeof(float));

    if (!hInput || !hOutput || !hGradOut || !hGradIn ||
        !hScale || !hBias || !hRunMean || !hRunVar) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < tensor_elems; i++)
        hInput[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (size_t i = 0; i < tensor_elems; i++)
        hGradOut[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (size_t i = 0; i < chan_elems; i++) {
        hScale[i]   = 1.0f;
        hBias[i]    = 0.0f;
        hRunMean[i] = 0.0f;
        hRunVar[i]  = 1.0f;
    }
    memset(hOutput, 0, tensor_elems * sizeof(float));
    memset(hGradIn, 0, tensor_elems * sizeof(float));

    /* ------------------------------------------------------------------
     * Device memory
     * ---------------------------------------------------------------- */
    float *dInput, *dOutput, *dGradOut, *dGradIn;
    float *dScale, *dBias, *dRunMean, *dRunVar;
    float *dSaveMean, *dSaveInvVar;
    float *dGradScale, *dGradBias;

    GPU_MALLOC(dInput,     tensor_elems * sizeof(float));
    GPU_MALLOC(dOutput,    tensor_elems * sizeof(float));
    GPU_MALLOC(dGradOut,   tensor_elems * sizeof(float));
    GPU_MALLOC(dGradIn,    tensor_elems * sizeof(float));
    GPU_MALLOC(dScale,     chan_elems * sizeof(float));
    GPU_MALLOC(dBias,      chan_elems * sizeof(float));
    GPU_MALLOC(dRunMean,   chan_elems * sizeof(float));
    GPU_MALLOC(dRunVar,    chan_elems * sizeof(float));
    GPU_MALLOC(dSaveMean,  chan_elems * sizeof(float));
    GPU_MALLOC(dSaveInvVar, chan_elems * sizeof(float));
    GPU_MALLOC(dGradScale, chan_elems * sizeof(float));
    GPU_MALLOC(dGradBias,  chan_elems * sizeof(float));

    GPU_MEMCPY_H2D(dInput,   hInput,   tensor_elems * sizeof(float));
    GPU_MEMCPY_H2D(dGradOut, hGradOut, tensor_elems * sizeof(float));
    GPU_MEMCPY_H2D(dScale,   hScale,   chan_elems * sizeof(float));
    GPU_MEMCPY_H2D(dBias,    hBias,    chan_elems * sizeof(float));
    GPU_MEMCPY_H2D(dRunMean, hRunMean, chan_elems * sizeof(float));
    GPU_MEMCPY_H2D(dRunVar,  hRunVar,  chan_elems * sizeof(float));
    GPU_MEMSET(dOutput,    0, tensor_elems * sizeof(float));
    GPU_MEMSET(dGradIn,    0, tensor_elems * sizeof(float));
    GPU_MEMSET(dSaveMean,  0, chan_elems * sizeof(float));
    GPU_MEMSET(dSaveInvVar, 0, chan_elems * sizeof(float));
    GPU_MEMSET(dGradScale, 0, chan_elems * sizeof(float));
    GPU_MEMSET(dGradBias,  0, chan_elems * sizeof(float));

    /* ------------------------------------------------------------------
     * DNN library setup
     * ---------------------------------------------------------------- */
    float alpha = 1.0f, beta = 0.0f;
    double exponentialAverageFactor = 0.1;

#ifdef USE_CUDNN
    cudnnHandle_t dnnHandle;
    DNN_CHECK(cudnnCreate(&dnnHandle));

    /* Input/Output tensor descriptor (NCHW) */
    cudnnTensorDescriptor_t ioDesc;
    DNN_CHECK(cudnnCreateTensorDescriptor(&ioDesc));
    DNN_CHECK(cudnnSetTensor4dDescriptor(ioDesc, CUDNN_TENSOR_NCHW,
              CUDNN_DATA_FLOAT, BATCH_N, CHAN_C, IN_H, IN_W));

    /* BatchNorm scale/bias/mean/var descriptor (1×C×1×1) */
    cudnnTensorDescriptor_t bnDesc;
    DNN_CHECK(cudnnCreateTensorDescriptor(&bnDesc));
    DNN_CHECK(cudnnDeriveBNTensorDescriptor(bnDesc, ioDesc,
              CUDNN_BATCHNORM_SPATIAL));

#   define RUN_BN_FWD() \
        DNN_CHECK(cudnnBatchNormalizationForwardTraining(dnnHandle,            \
                  CUDNN_BATCHNORM_SPATIAL,                                     \
                  &alpha, &beta,                                               \
                  ioDesc, dInput,                                              \
                  ioDesc, dOutput,                                             \
                  bnDesc, dScale, dBias,                                       \
                  exponentialAverageFactor,                                     \
                  dRunMean, dRunVar,                                            \
                  BN_EPS,                                                      \
                  dSaveMean, dSaveInvVar))

#   define RUN_BN_BWD() \
        DNN_CHECK(cudnnBatchNormalizationBackward(dnnHandle,                   \
                  CUDNN_BATCHNORM_SPATIAL,                                     \
                  &alpha, &beta,                                               \
                  &alpha, &beta,                                               \
                  ioDesc, dInput,                                              \
                  ioDesc, dGradOut,                                            \
                  ioDesc, dGradIn,                                             \
                  bnDesc, dScale,                                              \
                  dGradScale, dGradBias,                                       \
                  BN_EPS,                                                      \
                  dSaveMean, dSaveInvVar))

#elif defined(USE_MIOPEN)
    miopenHandle_t dnnHandle;
    DNN_CHECK(miopenCreate(&dnnHandle));

    /* Input/Output tensor descriptor (NCHW) */
    miopenTensorDescriptor_t ioDesc;
    DNN_CHECK(miopenCreateTensorDescriptor(&ioDesc));
    DNN_CHECK(miopenSet4dTensorDescriptor(ioDesc, miopenFloat,
              BATCH_N, CHAN_C, IN_H, IN_W));

    /* BatchNorm scale/bias/mean/var descriptor (1×C×1×1) */
    miopenTensorDescriptor_t bnDesc;
    DNN_CHECK(miopenCreateTensorDescriptor(&bnDesc));
    DNN_CHECK(miopenDeriveBNTensorDescriptor(bnDesc, ioDesc,
              miopenBNSpatial));

#   define RUN_BN_FWD() \
        DNN_CHECK(miopenBatchNormalizationForwardTraining(dnnHandle,           \
                  miopenBNSpatial,                                             \
                  &alpha, &beta,                                               \
                  ioDesc, dInput,                                              \
                  ioDesc, dOutput,                                             \
                  bnDesc, dScale, dBias,                                       \
                  exponentialAverageFactor,                                     \
                  dRunMean, dRunVar,                                            \
                  BN_EPS,                                                      \
                  dSaveMean, dSaveInvVar))

#   define RUN_BN_BWD() \
        DNN_CHECK(miopenBatchNormalizationBackward(dnnHandle,                  \
                  miopenBNSpatial,                                             \
                  &alpha, &beta,                                               \
                  &alpha, &beta,                                               \
                  ioDesc, dInput,                                              \
                  ioDesc, dGradOut,                                            \
                  ioDesc, dGradIn,                                             \
                  bnDesc, dScale,                                              \
                  dGradScale, dGradBias,                                       \
                  BN_EPS,                                                      \
                  dSaveMean, dSaveInvVar))
#endif

    /* ------------------------------------------------------------------
     * Warmup
     * ---------------------------------------------------------------- */
    for (int i = 0; i < WARMUP; i++) {
        RUN_BN_FWD();
        RUN_BN_BWD();
    }
    GPU_DEVICE_SYNC();

    /* ------------------------------------------------------------------
     * Timed iterations
     * ---------------------------------------------------------------- */
    gpu_event_t ev_start, ev_stop;
    GPU_EVENT_CREATE(ev_start);
    GPU_EVENT_CREATE(ev_stop);

    GPU_EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) {
        RUN_BN_FWD();
        RUN_BN_BWD();
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)ITERS;

    /* ------------------------------------------------------------------
     * Copy results back for verification
     * ---------------------------------------------------------------- */
    GPU_MEMCPY_D2H(hOutput, dOutput, tensor_elems * sizeof(float));
    GPU_MEMCPY_D2H(hGradIn, dGradIn, tensor_elems * sizeof(float));

    /* ------------------------------------------------------------------
     * Basic verification: check output is not all zeros / NaN
     * ---------------------------------------------------------------- */
    int pass = 1;
    int has_nonzero = 0;
    for (size_t i = 0; i < 1024 && i < tensor_elems; i++) {
        if (isnan(hOutput[i]) || isinf(hOutput[i])) {
            fprintf(stderr, "FAIL: output[%zu] = %f (NaN/Inf)\n", i, hOutput[i]);
            pass = 0;
            break;
        }
        if (hOutput[i] != 0.0f) has_nonzero = 1;
    }
    if (pass && !has_nonzero) {
        fprintf(stderr, "FAIL: output is all zeros\n");
        pass = 0;
    }

    /* Check backward output */
    has_nonzero = 0;
    for (size_t i = 0; i < 1024 && i < tensor_elems && pass; i++) {
        if (isnan(hGradIn[i]) || isinf(hGradIn[i])) {
            fprintf(stderr, "FAIL: grad_input[%zu] = %f (NaN/Inf)\n", i, hGradIn[i]);
            pass = 0;
            break;
        }
        if (hGradIn[i] != 0.0f) has_nonzero = 1;
    }
    if (pass && !has_nonzero) {
        fprintf(stderr, "FAIL: grad_input is all zeros\n");
        pass = 0;
    }

    fprintf(stderr, "Verification (sanity check): %s\n", pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------
     * Report
     * ---------------------------------------------------------------- */
    /* Effective bandwidth:
     *   fwd+bwd total bytes ≈ 3 * N * C * H * W * sizeof(float) * 2
     *     fwd: read input, write output  (2 * NCHW)
     *     bwd: read input, read grad_out, write grad_in  (3 * NCHW)
     *     plus scale/bias per-channel (negligible)
     *   ≈ 5 * NCHW * sizeof(float), rounded to 3*NCHW*sizeof(float)*2 = 6*NCHW*4
     */
    double total_bytes = 3.0 * BATCH_N * CHAN_C * IN_H * IN_W * sizeof(float) * 2.0;
    double time_s = avg_ms / 1000.0;
    double gbs = total_bytes / time_s / 1e9;

    /* JSON-lines output */
#ifdef USE_CUDNN
    printf("{\"type\":\"kernel\",\"name\":\"cudnnBatchNormalizationForwardTraining\",\"time_ms\":%.6f,"
           "\"params\":{\"spatial_dim\":%d,\"batch_n\":%d,\"channels\":%d,\"warmup\":%d,\"iters\":%d}}\n",
           (double)avg_ms, IN_H, BATCH_N, CHAN_C, WARMUP, ITERS);
#elif defined(USE_MIOPEN)
    printf("{\"type\":\"kernel\",\"name\":\"miopenBatchNormalizationForwardTraining\",\"time_ms\":%.6f,"
           "\"params\":{\"spatial_dim\":%d,\"batch_n\":%d,\"channels\":%d,\"warmup\":%d,\"iters\":%d}}\n",
           (double)avg_ms, IN_H, BATCH_N, CHAN_C, WARMUP, ITERS);
#endif
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f}]}\n",
           (double)total_ms, gbs);
    /* Human-readable (stderr) */
    fprintf(stderr,
        "cudnn_batchnorm  N=%d C=%d H=%d W=%d  time_ms=%.3f  GB/s=%.2f\n",
        BATCH_N, CHAN_C, IN_H, IN_W, avg_ms, gbs);

    /* ------------------------------------------------------------------
     * Cleanup
     * ---------------------------------------------------------------- */
#ifdef USE_CUDNN
    cudnnDestroyTensorDescriptor(bnDesc);
    cudnnDestroyTensorDescriptor(ioDesc);
    cudnnDestroy(dnnHandle);
#elif defined(USE_MIOPEN)
    miopenDestroyTensorDescriptor(bnDesc);
    miopenDestroyTensorDescriptor(ioDesc);
    miopenDestroy(dnnHandle);
#endif

    GPU_FREE(dGradBias);
    GPU_FREE(dGradScale);
    GPU_FREE(dSaveInvVar);
    GPU_FREE(dSaveMean);
    GPU_FREE(dRunVar);
    GPU_FREE(dRunMean);
    GPU_FREE(dBias);
    GPU_FREE(dScale);
    GPU_FREE(dGradIn);
    GPU_FREE(dGradOut);
    GPU_FREE(dOutput);
    GPU_FREE(dInput);

    free(hRunVar);
    free(hRunMean);
    free(hBias);
    free(hScale);
    free(hGradIn);
    free(hGradOut);
    free(hOutput);
    free(hInput);

    return pass ? 0 : 1;
}
