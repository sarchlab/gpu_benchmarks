// Fixed PolyBench/GPU 3D convolution, independently mapped to one 3D grid.
// Operator pin: PolyBench-ACC 70ea4ca, CUDA/stencils/convolution-3d.
// The repeated k-1 offsets below are present in both pinned CPU and GPU code.
// One timed stencil launch; full measured-output verification follows timing.
#include "../../../common_cuda/bench_timing.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static bool read_int(const char* name, const char* text, int* value) {
    errno = 0;
    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 1 || parsed > INT_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        return false;
    }
    *value = (int)parsed;
    return true;
}

__global__ void conv3d_kernel(const float* __restrict__ input,
                             float* __restrict__ output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= N-1 || j >= N-1 || k >= N-1) return;
    const int plane = N*N;
    // Preserve all fifteen terms and their order, including repeated offsets.
    output[i*plane+j*N+k] =
          2.0f * input[(i-1)*plane+(j-1)*N+k-1]
        + 4.0f * input[(i+1)*plane+(j-1)*N+k-1]
        + 5.0f * input[(i-1)*plane+(j-1)*N+k-1]
        + 7.0f * input[(i+1)*plane+(j-1)*N+k-1]
        - 8.0f * input[(i-1)*plane+(j-1)*N+k-1]
        +10.0f * input[(i+1)*plane+(j-1)*N+k-1]
        - 3.0f * input[i*plane+(j-1)*N+k]
        + 6.0f * input[i*plane+j*N+k]
        - 9.0f * input[i*plane+(j+1)*N+k]
        + 2.0f * input[(i-1)*plane+(j-1)*N+k+1]
        + 4.0f * input[(i+1)*plane+(j-1)*N+k+1]
        + 5.0f * input[(i-1)*plane+j*N+k+1]
        + 7.0f * input[(i+1)*plane+j*N+k+1]
        - 8.0f * input[(i-1)*plane+(j+1)*N+k+1]
        +10.0f * input[(i+1)*plane+(j+1)*N+k+1];
}

static bool verify_convolution3d(const float* input, const float* output, int N) {
    // Independent CPU reference: combine the repeated offsets into eleven
    // spatial weights. Rounding scale retains the absolute weights of all
    // fifteen source terms (2 + 5 - 8 becomes -1, but its scale is 15).
    const int offsets[11][3] = {{-1,-1,-1},{1,-1,-1},{0,-1,0},{0,0,0},
        {0,1,0},{-1,-1,1},{1,-1,1},{-1,0,1},{1,0,1},{-1,1,1},{1,1,1}};
    const int weights[11] = {-1,21,-3,6,-9,2,4,5,7,-8,10};
    const int abs_weights[11] = {15,21,3,6,9,2,4,5,7,8,10};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                bool interior = i > 0 && j > 0 && k > 0 && i < N-1 && j < N-1 && k < N-1;
                double expected = 0.0, scale = 0.0;
                if (interior) {
                    for (int t = 0; t < 11; ++t) {
                        double x = input[(i+offsets[t][0])*N*N+(j+offsets[t][1])*N+k+offsets[t][2]];
                        expected += weights[t] * x;
                        scale += abs_weights[t] * std::fabs(x);
                    }
                }
                float actual = output[(i*N+j)*N+k];
                // ~33 float unit roundoffs times the source absolute-term sum.
                // Borders must remain exactly zero. No relative-only gate near cancellation.
                double tolerance = interior ? 2e-6 * std::max(1.0, scale) : 0.0;
                if (!std::isfinite(actual) || !std::isfinite(expected) ||
                    !std::isfinite(scale) || std::fabs((double)actual-expected) > tolerance) {
                    fprintf(stderr, "FAIL: measured convolution[%d,%d,%d]: got %.9g expected %.12g (tol %.9g)\n",
                            i,j,k,actual,expected,tolerance);
                    return false;
                }
            }
        }
    }
    fprintf(stderr, "PASS: verified all %d measured outputs (full_cpu_v1)\n", N*N*N);
    return true;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int N = 128, block_size = 8;
    if (getenv("BENCH_PARAM_filter_size")) {
        fprintf(stderr, "filter_size is unsupported: this benchmark uses the fixed PolyBench/GPU stencil\n");
        return EXIT_FAILURE;
    }
    for (int a = 1; a < argc; a += 2) {
        int* value = strcmp(argv[a], "--size") == 0 ? &N :
                     strcmp(argv[a], "--block_size") == 0 ? &block_size : nullptr;
        if (!value || a+1 >= argc) {
            fprintf(stderr, "Usage: %s [--size N] [--block_size B]; filter_size is unsupported\n", argv[0]);
            return EXIT_FAILURE;
        }
        if (!read_int(argv[a], argv[a+1], value)) return EXIT_FAILURE;
    }
    const char* env = getenv("BENCH_PARAM_size");
    if (env && !read_int("size", env, &N)) return EXIT_FAILURE;
    env = getenv("BENCH_PARAM_block_size");
    if (env && !read_int("block_size", env, &block_size)) return EXIT_FAILURE;
    if (N < 3 || (long long)N*N > INT_MAX/N) {
        fprintf(stderr, "size must be >= 3 and N^3 must fit signed int indexing\n");
        return EXIT_FAILURE;
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    if (block_size > prop.maxThreadsDim[0] || block_size > prop.maxThreadsDim[1] ||
        block_size > prop.maxThreadsDim[2] ||
        (long long)block_size*block_size > prop.maxThreadsPerBlock/block_size) {
        fprintf(stderr, "block_size^3 exceeds selected device launch limits\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s (id=%d)\nVolume: %dx%dx%d | fixed PolyBench/GPU stencil\n",
            prop.name,device_id,N,N,N);
    const size_t bytes = (size_t)N*N*N*sizeof(float);
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Host allocation failed\n");
        free(h_input); free(h_output);
        return EXIT_FAILURE;
    }
    // Keep the existing input sequence. Removing the later random filter
    // generation does not change these draws; cross-libc identity is R-27.
    srand(42);
    for (int n = 0; n < N*N*N; ++n) h_input[n] = (float)(rand()%100)/100.0f;
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes)); // defined output border; untimed setup
    dim3 block(block_size,block_size,block_size);
    dim3 grid((N+block_size-1)/block_size,(N+block_size-1)/block_size,(N+block_size-1)/block_size);
    char params[192];
    snprintf(params, sizeof(params), "\"size\":%d,\"block_size\":%d,\"operator\":\"polybench_gpu_70ea4ca\",\"verification\":\"full_cpu_v1\"", N,block_size);
    double total_gpu_ms = bench_run_once("conv3d_kernel", params, [&]() {
        conv3d_kernel<<<grid,block>>>(d_input,d_output,N);
    });
    bench_emit_summary(total_gpu_ms);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    const bool verified = verify_convolution3d(h_input,h_output,N);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input); free(h_output);
    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
