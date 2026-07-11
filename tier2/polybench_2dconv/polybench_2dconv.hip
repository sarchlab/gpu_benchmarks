// polybench_2dconv.hip — PolyBench 2D Convolution benchmark (HIP, self-contained)
//
// Applies a fixed 3×3 kernel to an NI×NJ matrix A, producing output B.
// Same 9-tap structure as PolyBench/GPU 2DCONV, but with a fixed coefficient
// set that differs from upstream's values (canonical PolyBench/GPU uses a
// signed c11..c33 set):
//
//   c = {{0.8, 0.2, 0.3},
//        {0.2, 0.7, 0.4},
//        {0.1, 0.2, 0.5}}
//
//   B[i][j] = sum_{k1=0}^{2} sum_{k2=0}^{2} c[k1][k2] * A[i+k1-1][j+k2-1]
//             for 1 <= i < NI-1, 1 <= j < NJ-1
//
// Each GPU thread computes one output element (one 3×3 dot product).
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./polybench_2dconv [--size N]
//
//   --size N        Matrix dimension N×N   (default: 2048)
//
// Environment variables (override argv):
//   BENCH_PARAM_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// launch, carrying the GPU event time (time_ms) and host wall-clock time
// (host_time_ms), then a summary whose total_time_ms equals that GPU event
// time. GFLOPS is derived downstream from time_ms and params.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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
// PolyBench 2D Convolution kernel
// ---------------------------------------------------------------------------

__global__ void convolution2D_kernel(const float* __restrict__ A,
                                     float*       __restrict__ B,
                                     int NI, int NJ) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (i < 1 || i >= NI - 1 || j < 1 || j >= NJ - 1)
        return;

    // Fixed coefficients (differ from upstream PolyBench/GPU's signed set)
    const float c00 = 0.8f, c01 = 0.2f, c02 = 0.3f;
    const float c10 = 0.2f, c11 = 0.7f, c12 = 0.4f;
    const float c20 = 0.1f, c21 = 0.2f, c22 = 0.5f;

    B[i * NJ + j] =
        c00 * A[(i - 1) * NJ + (j - 1)] +
        c01 * A[(i - 1) * NJ +  j     ] +
        c02 * A[(i - 1) * NJ + (j + 1)] +
        c10 * A[ i      * NJ + (j - 1)] +
        c11 * A[ i      * NJ +  j     ] +
        c12 * A[ i      * NJ + (j + 1)] +
        c20 * A[(i + 1) * NJ + (j - 1)] +
        c21 * A[(i + 1) * NJ +  j     ] +
        c22 * A[(i + 1) * NJ + (j + 1)];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N     = parseIntParam(argc, argv, "--size", 2048);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    int NI = N, NJ = N;
    size_t matSize = (size_t)NI * NJ * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix: %d×%d\n\n", NI, NJ);

    // Host allocations
    float* h_A = (float*)malloc(matSize);
    if (!h_A) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(42);
    for (int i = 0; i < NI * NJ; i++)
        h_A[i] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, matSize));
    HIP_CHECK(hipMalloc(&d_B, matSize));

    HIP_CHECK(hipMemcpy(d_A, h_A, matSize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B, 0, matSize));

    dim3 block(16, 16);
    dim3 grid((NJ + block.x - 1) / block.x, (NI + block.y - 1) / block.y);

    char params[160];
    snprintf(params, sizeof(params), "\"size\":%d", N);

    // Single timed launch (optionally preceded by the vector-add warmer).
    // Input A is read-only and output B is fully overwritten, so every
    // process execution performs identical work — no reset needed.
    double total_gpu_ms =
        bench_run_once("convolution2D_kernel", params, [&]() {
            convolution2D_kernel<<<grid, block>>>(d_A, d_B, NI, NJ);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_A);

    return 0;
}
