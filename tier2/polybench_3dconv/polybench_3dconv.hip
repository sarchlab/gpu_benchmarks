// polybench_3dconv.hip — PolyBench 3D Convolution benchmark (HIP, self-contained)
//
// 3D convolution over an NxNxN volume with a small 3D filter (e.g., 3x3x3).
// Each output point is the weighted sum of its neighborhood.
// Derived from the PolyBench/GPU benchmark suite.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_filter_size
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// carrying the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the SUM of all iterations' GPU event times. GFLOPS is
// derived downstream from time_ms and params.
// Output (stderr): device info, per-iteration timing

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
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
// 3D Convolution Kernel
// ---------------------------------------------------------------------------

__global__ void conv3d_kernel(const float* __restrict__ input,
                              const float* __restrict__ filter,
                              float* __restrict__ output,
                              int N, int filter_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= N || k >= N) return;

    int half = filter_size / 2;
    float sum = 0.0f;

    for (int fi = 0; fi < filter_size; ++fi) {
        int ii = i - half + fi;
        if (ii < 0 || ii >= N) continue;
        for (int fj = 0; fj < filter_size; ++fj) {
            int jj = j - half + fj;
            if (jj < 0 || jj >= N) continue;
            for (int fk = 0; fk < filter_size; ++fk) {
                int kk = k - half + fk;
                if (kk < 0 || kk >= N) continue;
                sum += input[((size_t)ii * N + jj) * N + kk]
                     * filter[((size_t)fi * filter_size + fj) * filter_size + fk];
            }
        }
    }

    output[((size_t)i * N + j) * N + k] = sum;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N           = parseIntParam(argc, argv, "--size", 128);
    int block_size  = parseIntParam(argc, argv, "--block_size", 8);
    int filter_size = parseIntParam(argc, argv, "--filter_size", 3);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_filter_size");
    if (env_val) filter_size = atoi(env_val);
    size_t vol_elems   = (size_t)N * N * N;
    size_t vol_bytes   = vol_elems * sizeof(float);
    size_t filt_elems  = (size_t)filter_size * filter_size * filter_size;
    size_t filt_bytes  = filt_elems * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Volume: %dx%dx%d  |  Filter: %dx%dx%d\n\n",
            N, N, N, filter_size, filter_size, filter_size);

    // Host allocations
    float* h_input  = (float*)malloc(vol_bytes);
    float* h_filter = (float*)malloc(filt_bytes);
    float* h_output = (float*)malloc(vol_bytes);

    // Synthetic data init
    srand(42);
    for (size_t i = 0; i < vol_elems; ++i)
        h_input[i] = (float)(rand() % 100) / 100.0f;
    float filt_sum = 0.0f;
    for (size_t i = 0; i < filt_elems; ++i) {
        h_filter[i] = (float)(rand() % 10 + 1);
        filt_sum += h_filter[i];
    }
    for (size_t i = 0; i < filt_elems; ++i)
        h_filter[i] /= filt_sum;

    // Device allocations
    float *d_input, *d_filter, *d_output;
    HIP_CHECK(hipMalloc(&d_input,  vol_bytes));
    HIP_CHECK(hipMalloc(&d_filter, filt_bytes));
    HIP_CHECK(hipMalloc(&d_output, vol_bytes));

    HIP_CHECK(hipMemcpy(d_input,  h_input,  vol_bytes,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_filter, h_filter, filt_bytes, hipMemcpyHostToDevice));

    dim3 block(block_size, block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size,
              (N + block_size - 1) / block_size,
              (N + block_size - 1) / block_size);

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"filter_size\":%d",
             N, block_size, filter_size);

    // Warmup + timed iterations, each recorded individually. The kernel
    // rewrites d_output as a pure function of the constant input volume and
    // filter, so repeated launches do identical work — no reset needed.
    double total_gpu_ms =
        bench_run_once("conv3d_kernel", params, [&]() {
            conv3d_kernel<<<grid, block>>>(d_input, d_filter, d_output, N,
                                           filter_size);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_filter));
    HIP_CHECK(hipFree(d_output));
    free(h_input);
    free(h_filter);
    free(h_output);

    return 0;
}
