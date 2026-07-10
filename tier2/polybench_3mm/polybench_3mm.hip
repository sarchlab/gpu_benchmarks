// polybench_3mm.hip — PolyBench 3mm benchmark (HIP, self-contained)
//
// Three chained matrix multiplications:
//   E = A * B  (NI×NK · NK×NJ → NI×NJ)
//   F = C * D  (NJ×NM · NM×NL → NJ×NL)
//   G = E * F  (NI×NJ · NJ×NL → NI×NL)
//
// Default matrix sizes: NI=NJ=NK=NL=NM=256 (all square 256×256)
//
// Usage:
//   ./polybench_3mm [--size N]
//
//   --size N         Matrix dimension N (NxN for all matrices) (default: 256)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_precision
//   BENCH_PARAM_warmup   (warmup passes, default 3)
//   BENCH_PARAM_iteration    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (mm3_kernel1/2/3) carrying that kernel's GPU event time and host
// wall-clock time within the pass, then a summary whose total_time_ms is the
// SUM of all recorded GPU event times. GFLOPS
// (total_flops = 2*NI*NK*NJ + 2*NJ*NM*NL + 2*NI*NJ*NL) is derived downstream.
// Output (stderr): device info, per-pass timing

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
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
// GPU Kernels
// Each thread computes one output element (simple dot-product loop).
// Block: 16×16, Grid: ceil(cols/16) × ceil(rows/16)
// ---------------------------------------------------------------------------

// Kernel 1: E = A * B   (A: NI×NK, B: NK×NJ → E: NI×NJ)
__global__ void mm3_kernel1(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__       E,
                             int NI, int NK, int NJ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && j < NJ) {
        float sum = 0.0f;
        for (int k = 0; k < NK; k++)
            sum += A[i * NK + k] * B[k * NJ + j];
        E[i * NJ + j] = sum;
    }
}

// Kernel 2: F = C * D   (C: NJ×NM, D: NM×NL → F: NJ×NL)
__global__ void mm3_kernel2(const float* __restrict__ C,
                             const float* __restrict__ D,
                             float* __restrict__       F,
                             int NJ, int NM, int NL) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < NJ && l < NL) {
        float sum = 0.0f;
        for (int m = 0; m < NM; m++)
            sum += C[j * NM + m] * D[m * NL + l];
        F[j * NL + l] = sum;
    }
}

// Kernel 3: G = E * F   (E: NI×NJ, F: NJ×NL → G: NI×NL)
__global__ void mm3_kernel3(const float* __restrict__ E,
                             const float* __restrict__ F,
                             float* __restrict__       G,
                             int NI, int NJ, int NL) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && l < NL) {
        float sum = 0.0f;
        for (int j = 0; j < NJ; j++)
            sum += E[i * NJ + j] * F[j * NL + l];
        G[i * NL + l] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N     = parseIntParam(argc, argv, "--size", 256);
    int block_size = 256;
    const char* precision = "float";

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) precision = env_val;
    // Validate string params so the values echoed into the JSON records are
    // always canonical literals (never raw env text).
    if (strcmp(precision, "half") != 0 && strcmp(precision, "float") != 0 &&
        strcmp(precision, "double") != 0) {
        fprintf(stderr, "Invalid precision '%s' (use half, float, double)\n", precision);
        return EXIT_FAILURE;
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // All matrices are N×N
    const int NI = N, NJ = N, NK = N, NL = N, NM = N;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d (all)\n\n",
            N, N);

    // Buffer sizes
    size_t sA = (size_t)NI * NK * sizeof(float); // A: NI×NK
    size_t sB = (size_t)NK * NJ * sizeof(float); // B: NK×NJ
    size_t sC = (size_t)NJ * NM * sizeof(float); // C: NJ×NM
    size_t sD = (size_t)NM * NL * sizeof(float); // D: NM×NL
    size_t sE = (size_t)NI * NJ * sizeof(float); // E: NI×NJ
    size_t sF = (size_t)NJ * NL * sizeof(float); // F: NJ×NL
    size_t sG = (size_t)NI * NL * sizeof(float); // G: NI×NL

    // Host allocations — inputs only (A, B, C, D)
    float* h_A = (float*)malloc(sA);
    float* h_B = (float*)malloc(sB);
    float* h_C = (float*)malloc(sC);
    float* h_D = (float*)malloc(sD);

    // Initialize with random floats
    srand(42);
    for (int i = 0; i < NI * NK; ++i) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NK * NJ; ++i) h_B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NJ * NM; ++i) h_C[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NM * NL; ++i) h_D[i] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_G;
    HIP_CHECK(hipMalloc(&d_A, sA));
    HIP_CHECK(hipMalloc(&d_B, sB));
    HIP_CHECK(hipMalloc(&d_C, sC));
    HIP_CHECK(hipMalloc(&d_D, sD));
    HIP_CHECK(hipMalloc(&d_E, sE));
    HIP_CHECK(hipMalloc(&d_F, sF));
    HIP_CHECK(hipMalloc(&d_G, sG));

    HIP_CHECK(hipMemcpy(d_A, h_A, sA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, sB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C, sC, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_D, h_D, sD, hipMemcpyHostToDevice));

    // Kernel launch configuration (16×16 blocks)
    dim3 block(16, 16);
    // Kernel 1: E = A*B  grid over (NJ, NI)
    dim3 grid1((NJ + 15) / 16, (NI + 15) / 16);
    // Kernel 2: F = C*D  grid over (NL, NJ)
    dim3 grid2((NL + 15) / 16, (NJ + 15) / 16);
    // Kernel 3: G = E*F  grid over (NL, NI)
    dim3 grid3((NL + 15) / 16, (NI + 15) / 16);

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"precision\":\"%s\"",
             N, block_size, precision);

    // Per-kernel accumulators: each pass launches the three kernels once.
    // No state reset is needed between passes — inputs A/B/C/D are read-only
    // and E/F/G are fully overwritten every pass, so every pass does
    // identical work on identical data.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2, acc_k3;

    auto run_3mm_pass = [&]() {
        bench_lap_reset(&acc_k1);
        bench_lap_reset(&acc_k2);
        bench_lap_reset(&acc_k3);

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
            mm3_kernel1<<<grid1, block>>>(d_A, d_B, d_E, NI, NK, NJ);
        }));
        bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
            mm3_kernel2<<<grid2, block>>>(d_C, d_D, d_F, NJ, NM, NL);
        }));
        bench_lap_add(&acc_k3, bench_time_op(&timer, [&]() {
            mm3_kernel3<<<grid3, block>>>(d_E, d_F, d_G, NI, NJ, NL);
        }));
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_3mm_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_3mm_pass();
        bench_emit_iter("mm3_kernel1", i, &acc_k1, params);
        bench_emit_iter("mm3_kernel2", i, &acc_k2, params);
        bench_emit_iter("mm3_kernel3", i, &acc_k3, params);
        total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms + acc_k3.gpu_ms;
        fprintf(stderr, "pass %d: k1 gpu = %.3f ms, k2 gpu = %.3f ms, k3 gpu = %.3f ms\n",
                i, acc_k1.gpu_ms, acc_k2.gpu_ms, acc_k3.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_E));
    HIP_CHECK(hipFree(d_F));
    HIP_CHECK(hipFree(d_G));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}
