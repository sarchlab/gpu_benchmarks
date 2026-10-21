// shoc_reduction.cu — SHOC Reduction benchmark (CUDA, self-contained)
//
// Parallel sum reduction: reduces an array of floats to a single sum using
// tree-based shared-memory reduction. Derived from the SHOC benchmark suite.
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback.
//
// Environment variables:
//   BENCH_PARAM_size, BENCH_PARAM_block_size
//
// Output (stdout): for the single timed run (the full multi-level
// reduction), one {"type":"kernel","name":"reduce_kernel",...} JSON line
// carrying the run's accumulated GPU event time (time_ms) and host
// wall-clock time (host_time_ms) over all reduce_kernel launches, then a
// summary whose total_time_ms equals that record's GPU event time.
// Effective bandwidth (GB/s) is derived downstream from time_ms and params.
// Output (stderr): Human-readable diagnostics

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Inlined bench_common_hip.h utilities
// ---------------------------------------------------------------------------

// Error-checking macro
#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// Parse --<name> N from argv (returns defaultVal if not found)
static int parseIntParam(int argc, char** argv, const char* name,
                         int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Reduction kernel — shared-memory tree reduction
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (i < (unsigned int)N)
        val = input[i];
    if (i + blockDim.x < (unsigned int)N)
        val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Multi-level GPU reduction. Every reduce_kernel launch is individually
// bracketed with GPU events when a timer/accumulator pair is supplied
// (device-to-device copies stay outside the timed brackets); passing
// nullptrs runs untimed. `d_temp` is a pre-allocated scratch buffer for
// intermediate passes (may be nullptr when the first pass yields one block).
// ---------------------------------------------------------------------------
static void gpu_reduce(const float* d_input, float* d_partial, float* d_temp,
                       float* d_result, int N, int block_size,
                       bench_timer_t* timer, bench_lap_t* acc)
{
    int elems_per_block = block_size * 2;
    int num_blocks      = (N + elems_per_block - 1) / elems_per_block;
    int shared_bytes    = block_size * sizeof(float);

    auto timed_launch = [&](int grid, const float* src, float* dst, int n) {
        if (timer) {
            bench_lap_add(acc, bench_time_op(timer, [&]() {
                reduce_kernel<<<dim3(grid), dim3(block_size), shared_bytes, 0>>>(src, dst, n);
            }));
        } else {
            reduce_kernel<<<dim3(grid), dim3(block_size), shared_bytes, 0>>>(src, dst, n);
        }
    };

    timed_launch(num_blocks, d_input, d_partial, N);

    if (num_blocks == 1) {
        CUDA_CHECK(cudaMemcpy(d_result, d_partial, sizeof(float),
                             cudaMemcpyDeviceToDevice));
        return;
    }

    // Ping-pong between d_partial and d_temp so a level never reads and writes
    // the same buffer: reduce_kernel has each block read a strided span of `src`,
    // so an in-place launch (src==dst) with >1 output block would let block 0
    // read d_temp[1] while block 1 writes it. Both buffers hold the largest
    // intermediate this level or later needs (d_partial: num_blocks;
    // d_temp: the second-level count), and outputs only shrink, so capacity holds.
    int remaining = num_blocks;
    float* d_src  = d_partial;
    float* d_dst  = d_temp;

    while (remaining > 1) {
        int next_blocks  = (remaining + elems_per_block - 1) / elems_per_block;

        if (next_blocks == 1) {
            timed_launch(1, d_src, d_result, remaining);
            remaining = 1;
        } else {
            timed_launch(next_blocks, d_src, d_dst, remaining);
            float* tmp = d_src; d_src = d_dst; d_dst = tmp;
            remaining  = next_blocks;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
static double reduction_cpu(const float* data, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int size       = 4194304;
    int block_size = 256;

    // Command-line fallback
    size       = parseIntParam(argc, argv, "--size", size);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    int N     = size;
    size_t bytesIn = (size_t)N * sizeof(float);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d floats (%.1f MiB)\n",
            N, (double)bytesIn / (1024.0 * 1024.0));
    fprintf(stderr, "Block size: %d\n\n", block_size);

    // Host allocation
    float* h_input = (float*)malloc(bytesIn);

    // Initialize with a known pattern
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i % 1000) * 0.001f;
    }

    // Device allocations
    int elems_per_block = block_size * 2;
    int num_blocks      = (N + elems_per_block - 1) / elems_per_block;

    float *d_input, *d_partial, *d_result;
    CUDA_CHECK(cudaMalloc(&d_input,   bytesIn));
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result,  sizeof(float)));

    // Scratch buffer for intermediate reduction passes, allocated once here
    // so no allocation happens on the measured path.
    float* d_temp = nullptr;
    if (num_blocks > 1) {
        int temp_blocks = (num_blocks + elems_per_block - 1) / elems_per_block;
        CUDA_CHECK(cudaMalloc(&d_temp, (size_t)temp_blocks * sizeof(float)));
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytesIn, cudaMemcpyHostToDevice));

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d",
             size, block_size);

    // Accumulator over all reduce_kernel launches in the single full
    // multi-level reduction. The input array is never mutated and the
    // partial/result buffers are fully overwritten, so every execution
    // performs identical work — no state reset is needed.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_reduce;

    // Timed pass, recorded once
    double total_gpu_ms = 0.0;
    bench_lap_reset(&acc_reduce);
    /* Cold-cache calibration: evict the LLC and hold the clock high before
     * the pass's first timed launch, so it runs cold. No-op when
     * BENCH_PARAM_vector_add_warmup is disabled. */
    bench_pass_warm();

    gpu_reduce(d_input, d_partial, d_temp, d_result, N, block_size,
               &timer, &acc_reduce);
    bench_emit_iter("reduce_kernel", 0, &acc_reduce, params);
    total_gpu_ms += acc_reduce.gpu_ms;
    fprintf(stderr, "gpu = %.3f ms, host = %.3f ms\n",
            acc_reduce.gpu_ms, acc_reduce.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify
    float gpu_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_result, d_result, sizeof(float),
                         cudaMemcpyDeviceToHost));
    double cpu_result = reduction_cpu(h_input, N);

    double rel_err = fabs((double)gpu_result - cpu_result)
                     / (fabs(cpu_result) + 1e-10);
    fprintf(stderr, "GPU sum: %f, CPU sum: %f, relative error: %e\n",
            gpu_result, cpu_result, rel_err);

    const bool pass = std::isfinite(gpu_result) && std::isfinite(cpu_result) &&
                      std::isfinite(rel_err) && rel_err < 1e-3;
    if (pass) {
        fprintf(stderr, "PASS\n");
    } else {
        fprintf(stderr, "FAIL: relative error too large\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));
    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
    free(h_input);

    return pass ? 0 : 1;
}
