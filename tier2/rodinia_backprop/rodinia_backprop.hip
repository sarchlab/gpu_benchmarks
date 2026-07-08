/**
 * rodinia_backprop.hip — HIP/CUDA Backpropagation benchmark (Rodinia suite).
 *
 * Implements a two-layer fully-connected neural network forward + backward pass.
 * Architecture: INPUT_N → HIDDEN_N → OUTPUT_N
 *
 * Default sizes:
 *   INPUT_N  = 65536
 *   HIDDEN_N = 1024
 *   OUTPUT_N = 1
 *
 * Kernels:
 *   1. forward_hidden  — hidden[j] = sigmoid(sum_i input[i]*w1[i*HIDDEN_N+j] + b1[j])
 *   2. forward_output  — output[k] = sigmoid(sum_j hidden[j]*w2[j*OUTPUT_N+k] + b2[k])
 *   3. backward_output_delta — delta_out[k] = output[k]*(1-output[k])*(target[k]-output[k])
 *   4. backward_hidden_delta — delta_hid[j] = hidden[j]*(1-hidden[j])*sum_k(w2[j*OUTPUT_N+k]*delta_out[k])
 *   5. update_w1       — w1[i*HIDDEN_N+j] += lr * input[i] * delta_hid[j]
 *   6. update_w2       — w2[j*OUTPUT_N+k] += lr * hidden[j] * delta_out[k]
 *
 * Usage:
 *   ./rodinia_backprop [--input N] [--hidden H] [--output O]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_input, BENCH_PARAM_hidden, BENCH_PARAM_output
 *   BENCH_PARAM_num_warmup   (warmup epochs, default 3)
 *   BENCH_PARAM_num_iters    (timed epochs, each recorded, default 7)
 *
 * Output (stdout): per timed epoch, one {"type":"kernel",...} JSON line per
 * kernel carrying that kernel's GPU event time and host time within the
 * epoch, then a summary whose total_time_ms is the SUM of all recorded GPU
 * event times. GFLOPS is derived downstream from time_ms and params.
 * Output (stderr): device info, timing details
 *
 * Build:
 *   make PLATFORM=cuda   # NVIDIA
 *   make PLATFORM=rocm   # AMD
 */

/* Shared timing harness. Under nvcc the hip* names used by the harness are
 * rewritten to cuda* by hip_cuda_compat.h; the benchmark's own gpu* macro
 * layer below stays for its non-timing calls. */
#include "../../common/bench_timing.h"

#ifdef __HIP__
  #define GPU_API hip
  #define gpuMalloc       hipMalloc
  #define gpuFree         hipFree
  #define gpuMemcpy       hipMemcpy
  #define gpuMemcpyHtoD   hipMemcpyHostToDevice
  #define gpuMemcpyDtoH   hipMemcpyDeviceToHost
  #define gpuDeviceSynchronize hipDeviceSynchronize
  #define gpuGetDeviceProperties hipGetDeviceProperties
  typedef hipDeviceProp_t gpuDeviceProp_t;
#else
  #include <cuda_runtime.h>
  #define gpuMalloc       cudaMalloc
  #define gpuFree         cudaFree
  #define gpuMemcpy       cudaMemcpy
  #define gpuMemcpyHtoD   cudaMemcpyHostToDevice
  #define gpuMemcpyDtoH   cudaMemcpyDeviceToHost
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  typedef cudaDeviceProp gpuDeviceProp_t;
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Default network dimensions
// ---------------------------------------------------------------------------
#define DEFAULT_INPUT_N  65536
#define DEFAULT_HIDDEN_N 1024
#define DEFAULT_OUTPUT_N 1
#define DEFAULT_ITERS    5
#define LR               0.1f
#define BLOCK_SZ         256

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid_d(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * forward_hidden: compute activations for the hidden layer.
 * One thread per hidden unit j.
 */
__global__ void forward_hidden(const float* __restrict__ input,
                                const float* __restrict__ w1,
                                const float* __restrict__ b1,
                                float* __restrict__ hidden,
                                int input_n, int hidden_n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden_n) return;
    float sum = b1[j];
    for (int i = 0; i < input_n; i++)
        sum += input[i] * w1[i * hidden_n + j];
    hidden[j] = sigmoid_d(sum);
}

/**
 * forward_output: compute activations for the output layer.
 * One thread per output unit k.
 */
__global__ void forward_output(const float* __restrict__ hidden,
                                const float* __restrict__ w2,
                                const float* __restrict__ b2,
                                float* __restrict__ output,
                                int hidden_n, int output_n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= output_n) return;
    float sum = b2[k];
    for (int j = 0; j < hidden_n; j++)
        sum += hidden[j] * w2[j * output_n + k];
    output[k] = sigmoid_d(sum);
}

/**
 * backward_output_delta: compute error signal at output layer.
 * delta_out[k] = output[k] * (1 - output[k]) * (target[k] - output[k])
 */
__global__ void backward_output_delta(const float* __restrict__ output,
                                       const float* __restrict__ target,
                                       float* __restrict__ delta_out,
                                       int output_n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= output_n) return;
    float o = output[k];
    delta_out[k] = o * (1.0f - o) * (target[k] - o);
}

/**
 * backward_hidden_delta: backpropagate error to hidden layer.
 * delta_hid[j] = hidden[j] * (1 - hidden[j]) * sum_k(w2[j*output_n+k] * delta_out[k])
 */
__global__ void backward_hidden_delta(const float* __restrict__ hidden,
                                       const float* __restrict__ w2,
                                       const float* __restrict__ delta_out,
                                       float* __restrict__ delta_hid,
                                       int hidden_n, int output_n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden_n) return;
    float h = hidden[j];
    float sum = 0.0f;
    for (int k = 0; k < output_n; k++)
        sum += w2[j * output_n + k] * delta_out[k];
    delta_hid[j] = h * (1.0f - h) * sum;
}

/**
 * update_w1: update weights from input to hidden layer.
 * w1[i*hidden_n+j] += lr * input[i] * delta_hid[j]
 * 2D grid: x→i (input), y→j (hidden)
 */
__global__ void update_w1(float* __restrict__ w1,
                            const float* __restrict__ input,
                            const float* __restrict__ delta_hid,
                            int input_n, int hidden_n, float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= input_n || j >= hidden_n) return;
    w1[i * hidden_n + j] += lr * input[i] * delta_hid[j];
}

/**
 * update_w2: update weights from hidden to output layer.
 * w2[j*output_n+k] += lr * hidden[j] * delta_out[k]
 */
__global__ void update_w2(float* __restrict__ w2,
                            const float* __restrict__ hidden,
                            const float* __restrict__ delta_out,
                            int hidden_n, int output_n, float lr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden_n) return;
    for (int k = 0; k < output_n; k++)
        w2[j * output_n + k] += lr * hidden[j] * delta_out[k];
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
static int parseIntParam(int argc, char** argv, const char* name, int def)
{
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
        }
    }
    return def;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int input_n  = DEFAULT_INPUT_N;
    int hidden_n = DEFAULT_HIDDEN_N;
    int output_n = DEFAULT_OUTPUT_N;

    // Command-line fallback
    input_n  = parseIntParam(argc, argv, "--input", input_n);
    hidden_n = parseIntParam(argc, argv, "--hidden", hidden_n);
    output_n = parseIntParam(argc, argv, "--output", output_n);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_input");
    if (env_val) input_n = atoi(env_val);
    env_val = getenv("BENCH_PARAM_hidden");
    if (env_val) hidden_n = atoi(env_val);
    env_val = getenv("BENCH_PARAM_output");
    if (env_val) output_n = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Print device info
    gpuDeviceProp_t prop;
    gpuGetDeviceProperties(&prop, 0);
    fprintf(stderr, "Device: %s\n", prop.name);
    fprintf(stderr, "Network: %d → %d → %d  |  Iterations: %d warmup + %d timed\n\n",
            input_n, hidden_n, output_n, num_warmup, num_iters);

    // -----------------------------------------------------------------------
    // Host allocation and initialization
    // -----------------------------------------------------------------------
    size_t sz_w1  = (size_t)input_n  * hidden_n * sizeof(float);
    size_t sz_w2  = (size_t)hidden_n * output_n  * sizeof(float);
    size_t sz_in  = (size_t)input_n              * sizeof(float);
    size_t sz_hid = (size_t)hidden_n             * sizeof(float);
    size_t sz_out = (size_t)output_n             * sizeof(float);

    std::vector<float> h_input(input_n),  h_w1(input_n * hidden_n),
                       h_w2(hidden_n * output_n),
                       h_b1(hidden_n, 0.0f), h_b2(output_n, 0.0f),
                       h_target(output_n, 1.0f);

    srand(42);
    for (int i = 0; i < input_n; i++)
        h_input[i] = (float)(i % 256) / 256.0f;
    for (size_t i = 0; i < h_w1.size(); i++)
        h_w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;   // [-0.1, 0.1]
    for (size_t i = 0; i < h_w2.size(); i++)
        h_w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;

    // -----------------------------------------------------------------------
    // Device allocation
    // -----------------------------------------------------------------------
    float *d_input, *d_w1, *d_b1, *d_hidden;
    float *d_w2,    *d_b2, *d_output;
    float *d_target, *d_delta_out, *d_delta_hid;

    gpuMalloc((void**)&d_input,     sz_in);
    gpuMalloc((void**)&d_w1,        sz_w1);
    gpuMalloc((void**)&d_b1,        sz_hid);
    gpuMalloc((void**)&d_hidden,    sz_hid);
    gpuMalloc((void**)&d_w2,        sz_w2);
    gpuMalloc((void**)&d_b2,        sz_out);
    gpuMalloc((void**)&d_output,    sz_out);
    gpuMalloc((void**)&d_target,    sz_out);
    gpuMalloc((void**)&d_delta_out, sz_out);
    gpuMalloc((void**)&d_delta_hid, sz_hid);

    gpuMemcpy(d_input,  h_input.data(),  sz_in,  gpuMemcpyHtoD);
    gpuMemcpy(d_w1,     h_w1.data(),     sz_w1,  gpuMemcpyHtoD);
    gpuMemcpy(d_b1,     h_b1.data(),     sz_hid, gpuMemcpyHtoD);
    gpuMemcpy(d_w2,     h_w2.data(),     sz_w2,  gpuMemcpyHtoD);
    gpuMemcpy(d_b2,     h_b2.data(),     sz_out, gpuMemcpyHtoD);
    gpuMemcpy(d_target, h_target.data(), sz_out, gpuMemcpyHtoD);

    // -----------------------------------------------------------------------
    // Kernel launch configs
    // -----------------------------------------------------------------------
    // 1D kernels
    int grid_hid = (hidden_n + BLOCK_SZ - 1) / BLOCK_SZ;
    int grid_out = (output_n + BLOCK_SZ - 1) / BLOCK_SZ;

    // 2D kernel for update_w1
    dim3 block2d(16, 16);
    dim3 grid2d((input_n  + 15) / 16,
                (hidden_n + 15) / 16);

    char params[128];
    snprintf(params, sizeof(params),
             "\"input\":%d,\"hidden\":%d,\"output\":%d",
             input_n, hidden_n, output_n);

    // Per-kernel accumulators — each epoch launches every kernel once; each
    // kernel's GPU/host time within the epoch becomes that epoch's record.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_fwd_hid, acc_fwd_out, acc_bwd_out, acc_bwd_hid,
                acc_upd_w1, acc_upd_w2;

    // One forward + backward epoch. The weight updates mutate w1/w2 across
    // epochs, but every kernel runs fixed-trip-count loops with no
    // data-dependent control flow, so only floating-point VALUES drift —
    // the amount of work (and its timing) is value-independent. No reset.
    auto run_epoch = [&]() {
        bench_lap_reset(&acc_fwd_hid);
        bench_lap_reset(&acc_fwd_out);
        bench_lap_reset(&acc_bwd_out);
        bench_lap_reset(&acc_bwd_hid);
        bench_lap_reset(&acc_upd_w1);
        bench_lap_reset(&acc_upd_w2);

        // --- Forward ---
        bench_lap_add(&acc_fwd_hid, bench_time_op(&timer, [&]() {
            forward_hidden<<<grid_hid, BLOCK_SZ>>>(
                d_input, d_w1, d_b1, d_hidden, input_n, hidden_n);
        }));
        bench_lap_add(&acc_fwd_out, bench_time_op(&timer, [&]() {
            forward_output<<<grid_out, BLOCK_SZ>>>(
                d_hidden, d_w2, d_b2, d_output, hidden_n, output_n);
        }));

        // --- Backward ---
        bench_lap_add(&acc_bwd_out, bench_time_op(&timer, [&]() {
            backward_output_delta<<<grid_out, BLOCK_SZ>>>(
                d_output, d_target, d_delta_out, output_n);
        }));
        bench_lap_add(&acc_bwd_hid, bench_time_op(&timer, [&]() {
            backward_hidden_delta<<<grid_hid, BLOCK_SZ>>>(
                d_hidden, d_w2, d_delta_out, d_delta_hid, hidden_n, output_n);
        }));
        bench_lap_add(&acc_upd_w1, bench_time_op(&timer, [&]() {
            update_w1<<<grid2d, block2d>>>(
                d_w1, d_input, d_delta_hid, input_n, hidden_n, LR);
        }));
        bench_lap_add(&acc_upd_w2, bench_time_op(&timer, [&]() {
            update_w2<<<grid_hid, BLOCK_SZ>>>(
                d_w2, d_hidden, d_delta_out, hidden_n, output_n, LR);
        }));
    };

    // -----------------------------------------------------------------------
    // Warmup epochs (records not emitted), then timed epochs — one JSON
    // record per kernel per epoch.
    // -----------------------------------------------------------------------
    for (int w = 0; w < num_warmup; ++w) {
        run_epoch();
    }

    double total_gpu_ms = 0.0;
    for (int it = 0; it < num_iters; it++) {
        run_epoch();
        bench_emit_iter("forward_hidden",        it, &acc_fwd_hid, params);
        bench_emit_iter("forward_output",        it, &acc_fwd_out, params);
        bench_emit_iter("backward_output_delta", it, &acc_bwd_out, params);
        bench_emit_iter("backward_hidden_delta", it, &acc_bwd_hid, params);
        bench_emit_iter("update_w1",             it, &acc_upd_w1, params);
        bench_emit_iter("update_w2",             it, &acc_upd_w2, params);
        double epoch_gpu_ms = acc_fwd_hid.gpu_ms + acc_fwd_out.gpu_ms +
                              acc_bwd_out.gpu_ms + acc_bwd_hid.gpu_ms +
                              acc_upd_w1.gpu_ms + acc_upd_w2.gpu_ms;
        total_gpu_ms += epoch_gpu_ms;
        fprintf(stderr, "epoch %d: gpu = %.3f ms\n", it, epoch_gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d epochs\n",
            total_gpu_ms, num_iters);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    gpuFree(d_input);
    gpuFree(d_w1);
    gpuFree(d_b1);
    gpuFree(d_hidden);
    gpuFree(d_w2);
    gpuFree(d_b2);
    gpuFree(d_output);
    gpuFree(d_target);
    gpuFree(d_delta_out);
    gpuFree(d_delta_hid);

    return 0;
}
