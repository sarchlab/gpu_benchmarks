/**
 * cudnn_conv.hip — Forward 2D convolution benchmark using cuDNN (NVIDIA) or MIOpen (AMD).
 *
 * Computes forward convolution:
 *   Input:  N=32, C=3, H=W=224 (NCHW layout)
 *   Filter: K=64, C=3, R=S=7 (KCHW layout), stride=2, pad=3
 *   Output: N=32, K=64, H_out=W_out=112 (NCHW layout)
 *
 * Algorithm selection: auto-tune (choose fastest from available algorithms).
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations, report average
 *
 * Verification: compare a small patch of output against CPU reference
 * GFLOPS = 2 * N * K * H_out * W_out * C * R * S / time_s / 1e9
 *
 * Output:
 *   cudnn_conv,32x3x224x224,<time_ms>,<GFLOPS>
 *
 * Build:
 *   CUDA: nvcc -O3 -std=c++14 -x cu -DUSE_CUDNN  cudnn_conv.hip -lcudnn -o cudnn_conv
 *   ROCm: hipcc -O3 -std=c++14     -DUSE_MIOPEN cudnn_conv.hip -lMIOpen -o cudnn_conv
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
 * Benchmark parameters (runtime-configurable via BENCH_PARAM_* env vars)
 * ======================================================================= */
static int BATCH_N  = 32;
static int IN_C     = 3;
static int IN_H     = 224;
static int IN_W     = 224;
static int OUT_K    = 64;
static int FILT_R   = 7;
static int FILT_S   = 7;
static const int STRIDE = 2;
static const int PAD    = 3;
static int OUT_H    = 112;   /* floor((IN_H + 2*PAD - FILT_R) / STRIDE) + 1 */
static int OUT_W    = 112;
static const int WARMUP = 5;
static const int ITERS  = 5;

/* =========================================================================
 * CPU reference convolution — only for a small patch
 * Input  layout: NCHW  — input[((n*C + c)*H + h)*W + w]
 * Filter layout: KCHW  — filter[((k*C + c)*R + r)*S + s]
 * Output layout: NKHW  — output[((n*K + k)*OH + oh)*OW + ow]
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
                                int ih = oh * STRIDE - PAD + r;
                                int iw = ow * STRIDE - PAD + s;
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
 * main
 * ======================================================================= */
int main(void)
{
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

    /* Recompute output dimensions */
    OUT_H = (IN_H + 2 * PAD - FILT_R) / STRIDE + 1;
    OUT_W = (IN_W + 2 * PAD - FILT_S) / STRIDE + 1;

    /* ------------------------------------------------------------------
     * Host allocation and initialisation
     * ---------------------------------------------------------------- */
    size_t input_elems  = (size_t)BATCH_N * IN_C * IN_H * IN_W;
    size_t filter_elems = (size_t)OUT_K * IN_C * FILT_R * FILT_S;
    size_t output_elems = (size_t)BATCH_N * OUT_K * OUT_H * OUT_W;

    float *hInput  = (float *)malloc(input_elems  * sizeof(float));
    float *hFilter = (float *)malloc(filter_elems * sizeof(float));
    float *hOutput = (float *)malloc(output_elems * sizeof(float));

    if (!hInput || !hFilter || !hOutput) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < input_elems; i++)
        hInput[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (size_t i = 0; i < filter_elems; i++)
        hFilter[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    memset(hOutput, 0, output_elems * sizeof(float));

    /* ------------------------------------------------------------------
     * Device memory
     * ---------------------------------------------------------------- */
    float *dInput, *dFilter, *dOutput;
    GPU_MALLOC(dInput,  input_elems  * sizeof(float));
    GPU_MALLOC(dFilter, filter_elems * sizeof(float));
    GPU_MALLOC(dOutput, output_elems * sizeof(float));

    GPU_MEMCPY_H2D(dInput,  hInput,  input_elems  * sizeof(float));
    GPU_MEMCPY_H2D(dFilter, hFilter, filter_elems * sizeof(float));
    GPU_MEMSET(dOutput, 0, output_elems * sizeof(float));

    /* ------------------------------------------------------------------
     * DNN library setup
     * ---------------------------------------------------------------- */
    float alpha = 1.0f, beta = 0.0f;

#ifdef USE_CUDNN
    cudnnHandle_t dnnHandle;
    DNN_CHECK(cudnnCreate(&dnnHandle));

    /* Input tensor descriptor (NCHW) */
    cudnnTensorDescriptor_t inputDesc;
    DNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    DNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW,
              CUDNN_DATA_FLOAT, BATCH_N, IN_C, IN_H, IN_W));

    /* Output tensor descriptor (NCHW) */
    cudnnTensorDescriptor_t outputDesc;
    DNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    DNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW,
              CUDNN_DATA_FLOAT, BATCH_N, OUT_K, OUT_H, OUT_W));

    /* Filter descriptor (KCHW) */
    cudnnFilterDescriptor_t filterDesc;
    DNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    DNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
              CUDNN_TENSOR_NCHW, OUT_K, IN_C, FILT_R, FILT_S));

    /* Convolution descriptor */
    cudnnConvolutionDescriptor_t convDesc;
    DNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    DNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
              PAD, PAD, STRIDE, STRIDE, 1, 1,
              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    /* Auto-tune: find fastest algorithm */
    cudnnConvolutionFwdAlgoPerf_t perfResults[8];
    int returnedAlgoCount = 0;
    DNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(dnnHandle,
              inputDesc, filterDesc, convDesc, outputDesc,
              8, &returnedAlgoCount, perfResults));
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    fprintf(stderr, "cuDNN algo selected: %d (of %d candidates)\n",
            (int)algo, returnedAlgoCount);

    /* Workspace */
    size_t workspaceSize = 0;
    DNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(dnnHandle,
              inputDesc, filterDesc, convDesc, outputDesc,
              algo, &workspaceSize));
    void *dWorkspace = NULL;
    if (workspaceSize > 0) {
        GPU_MALLOC(dWorkspace, workspaceSize);
    }
    fprintf(stderr, "Workspace: %zu bytes\n", workspaceSize);

#   define RUN_CONV() \
        DNN_CHECK(cudnnConvolutionForward(dnnHandle, &alpha,            \
                  inputDesc, dInput, filterDesc, dFilter,               \
                  convDesc, algo, dWorkspace, workspaceSize,            \
                  &beta, outputDesc, dOutput))

#elif defined(USE_MIOPEN)
    miopenHandle_t dnnHandle;
    DNN_CHECK(miopenCreate(&dnnHandle));

    /* Input tensor descriptor (NCHW) */
    miopenTensorDescriptor_t inputDesc;
    DNN_CHECK(miopenCreateTensorDescriptor(&inputDesc));
    DNN_CHECK(miopenSet4dTensorDescriptor(inputDesc, miopenFloat,
              BATCH_N, IN_C, IN_H, IN_W));

    /* Output tensor descriptor (NCHW) */
    miopenTensorDescriptor_t outputDesc;
    DNN_CHECK(miopenCreateTensorDescriptor(&outputDesc));
    DNN_CHECK(miopenSet4dTensorDescriptor(outputDesc, miopenFloat,
              BATCH_N, OUT_K, OUT_H, OUT_W));

    /* Filter (weight) tensor descriptor — MIOpen uses tensor descriptor */
    miopenTensorDescriptor_t filterDesc;
    DNN_CHECK(miopenCreateTensorDescriptor(&filterDesc));
    DNN_CHECK(miopenSet4dTensorDescriptor(filterDesc, miopenFloat,
              OUT_K, IN_C, FILT_R, FILT_S));

    /* Convolution descriptor */
    miopenConvolutionDescriptor_t convDesc;
    DNN_CHECK(miopenCreateConvolutionDescriptor(&convDesc));
    DNN_CHECK(miopenInitConvolutionDescriptor(convDesc, miopenConvolution,
              PAD, PAD, STRIDE, STRIDE, 1, 1));

    /* Workspace size */
    size_t workspaceSize = 0;
    DNN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(dnnHandle,
              filterDesc, inputDesc, convDesc, outputDesc, &workspaceSize));
    void *dWorkspace = NULL;
    if (workspaceSize > 0) {
        GPU_MALLOC(dWorkspace, workspaceSize);
    }
    fprintf(stderr, "Workspace: %zu bytes\n", workspaceSize);

    /* Find best algorithm */
    miopenConvAlgoPerf_t miopenPerfResults[8];
    int returnedAlgoCount = 0;
    DNN_CHECK(miopenFindConvolutionForwardAlgorithm(dnnHandle,
              inputDesc, dInput, filterDesc, dFilter,
              convDesc, outputDesc, dOutput,
              8, &returnedAlgoCount, miopenPerfResults,
              dWorkspace, workspaceSize, false));
    miopenConvFwdAlgorithm_t algo = miopenPerfResults[0].fwd_algo;
    fprintf(stderr, "MIOpen algo selected: %d (of %d candidates)\n",
            (int)algo, returnedAlgoCount);

#   define RUN_CONV() \
        DNN_CHECK(miopenConvolutionForward(dnnHandle, &alpha,           \
                  inputDesc, dInput, filterDesc, dFilter,               \
                  convDesc, algo, &beta,                                \
                  outputDesc, dOutput, dWorkspace, workspaceSize))
#endif

    /* ------------------------------------------------------------------
     * Warmup
     * ---------------------------------------------------------------- */
    for (int i = 0; i < WARMUP; i++) {
        RUN_CONV();
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
        RUN_CONV();
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)ITERS;

    /* ------------------------------------------------------------------
     * Copy result back
     * ---------------------------------------------------------------- */
    GPU_MEMCPY_D2H(hOutput, dOutput, output_elems * sizeof(float));

    /* ------------------------------------------------------------------
     * Verify a small patch against CPU reference
     *   batch=0..0, k=0..3, oh=0..3, ow=0..3
     * ---------------------------------------------------------------- */
    const int VN = 1, VK = 4, VOH = 4, VOW = 4;
    size_t ref_elems = (size_t)VN * VK * VOH * VOW;
    float *cpuOut = (float *)calloc(ref_elems, sizeof(float));
    cpu_conv2d(VN, VK, VOH, VOW, hInput, hFilter, cpuOut);

    int pass = 1;
    for (int n = 0; n < VN && pass; n++) {
        for (int k = 0; k < VK && pass; k++) {
            for (int oh = 0; oh < VOH && pass; oh++) {
                for (int ow = 0; ow < VOW && pass; ow++) {
                    /* cpuOut uses compact layout: [n*VK+k][oh][ow] */
                    size_t ref_idx = ((size_t)n * VK + k) * VOH * VOW +
                                     (size_t)oh * VOW + ow;
                    /* hOutput uses full GPU layout: [n*OUT_K+k][OUT_H][OUT_W] */
                    size_t gpu_idx = ((size_t)n * OUT_K + k) * OUT_H * OUT_W +
                                     (size_t)oh * OUT_W + ow;
                    float ref = cpuOut[ref_idx];
                    float got = hOutput[gpu_idx];
                    float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
                    if (err > 1e-3f) {
                        fprintf(stderr,
                            "MISMATCH at [n=%d,k=%d,oh=%d,ow=%d]: "
                            "ref=%.6f got=%.6f rel=%.4e\n",
                            n, k, oh, ow, ref, got, err);
                        pass = 0;
                    }
                }
            }
        }
    }
    free(cpuOut);
    fprintf(stderr, "Verification (%d×%d×%d×%d patch): %s\n",
            VN, VK, VOH, VOW, pass ? "PASS" : "FAIL");

    /* ------------------------------------------------------------------
     * Report
     * ---------------------------------------------------------------- */
    double flops = 2.0 * BATCH_N * OUT_K * OUT_H * OUT_W *
                   IN_C * FILT_R * FILT_S;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines output */
#ifdef USE_CUDNN
    printf("{\"type\":\"kernel\",\"name\":\"cudnnConvolutionForward\",\"time_ms\":%.6f,"
           "\"params\":{\"input_hw\":%d,\"batch_n\":%d,\"in_channels\":%d,"
           "\"out_channels\":%d,\"filter_size\":%d}}\n",
           (double)avg_ms, IN_H, BATCH_N, IN_C, OUT_K, FILT_R);
#elif defined(USE_MIOPEN)
    printf("{\"type\":\"kernel\",\"name\":\"miopenConvolutionForward\",\"time_ms\":%.6f,"
           "\"params\":{\"input_hw\":%d,\"batch_n\":%d,\"in_channels\":%d,"
           "\"out_channels\":%d,\"filter_size\":%d}}\n",
           (double)avg_ms, IN_H, BATCH_N, IN_C, OUT_K, FILT_R);
#endif
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           (double)total_ms, gflops);

    /* Human-readable (stderr) */
    fprintf(stderr,
        "cudnn_conv  N=%d C=%d H=%d W=%d K=%d R=%d S=%d stride=%d pad=%d  "
        "time_ms=%.3f  GFLOPS=%.2f\n",
        BATCH_N, IN_C, IN_H, IN_W, OUT_K, FILT_R, FILT_S,
        STRIDE, PAD, avg_ms, gflops);

    /* ------------------------------------------------------------------
     * Cleanup
     * ---------------------------------------------------------------- */
#ifdef USE_CUDNN
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroy(dnnHandle);
#elif defined(USE_MIOPEN)
    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(filterDesc);
    miopenDestroyTensorDescriptor(outputDesc);
    miopenDestroyTensorDescriptor(inputDesc);
    miopenDestroy(dnnHandle);
#endif

    if (dWorkspace) GPU_FREE(dWorkspace);
    GPU_FREE(dOutput);
    GPU_FREE(dFilter);
    GPU_FREE(dInput);

    free(hOutput);
    free(hFilter);
    free(hInput);

    return pass ? 0 : 1;
}
