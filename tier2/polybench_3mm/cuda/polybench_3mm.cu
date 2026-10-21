// polybench_3mm.cu — PolyBench 3mm benchmark (CUDA, self-contained)
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
//   BENCH_PARAM_size
//
// Output (stdout): the 3mm pass runs exactly once; one {"type":"kernel",...}
// JSON line per kernel (mm3_kernel1/2/3) carrying that kernel's GPU event
// time and host wall-clock time, then a summary whose total_time_ms is the
// SUM of the three GPU event times. GFLOPS
// (total_flops = 2*NI*NK*NJ + 2*NJ*NM*NL + 2*NI*NJ*NL) is derived downstream.
// Output (stderr): device info, timing

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cpu_reference.h"

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    // All matrices are N×N
    const int NI = N, NJ = N, NK = N, NL = N, NM = N;

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
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
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));
    CUDA_CHECK(cudaMalloc(&d_D, sD));
    CUDA_CHECK(cudaMalloc(&d_E, sE));
    CUDA_CHECK(cudaMalloc(&d_F, sF));
    CUDA_CHECK(cudaMalloc(&d_G, sG));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, sC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, sD, cudaMemcpyHostToDevice));

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
             "\"size\":%d,\"verification\":\"full_cpu_v1\"", N);

    // Per-kernel accumulators: the single pass launches the three kernels
    // once each. No state reset is needed — inputs A/B/C/D are read-only and
    // E/F/G are fully overwritten, so every execution does identical work on
    // identical data.
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

    run_3mm_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("mm3_kernel1", 0, &acc_k1, params);
    bench_emit_iter("mm3_kernel2", 0, &acc_k2, params);
    bench_emit_iter("mm3_kernel3", 0, &acc_k3, params);
    total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms + acc_k3.gpu_ms;
    fprintf(stderr, "k1 gpu = %.3f ms, k2 gpu = %.3f ms, k3 gpu = %.3f ms\n",
            acc_k1.gpu_ms, acc_k2.gpu_ms, acc_k3.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify all three measured products after timing, with no extra GPU pass.
    const auto reference = cpu_3mm(h_A, h_B, h_C, h_D, NI, NJ, NK, NL, NM);
    const std::vector<double>* expected[] = {&reference.e, &reference.f, &reference.g};
    float* outputs[] = {d_E, d_F, d_G};
    const char* names[] = {"E", "F", "G"};
    size_t mismatches = 0, checked = 0;
    for (int stage = 0; stage < 3; ++stage) {
        // All dimensions are N here, so each output fits the original h_A
        // buffer after the complete independent reference has consumed it.
        CUDA_CHECK(cudaMemcpy(h_A, outputs[stage], expected[stage]->size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < expected[stage]->size(); ++i) {
            const double value = (*expected[stage])[i];
            if (!mm3_value_matches(h_A[i], value, stage == 2)) {
                if (mismatches < 10)
                    fprintf(stderr, "MISMATCH %s[%zu]: GPU=%.9g CPU=%.17g\n",
                            names[stage], i, h_A[i], value);
                ++mismatches;
            }
            ++checked;
        }
    }
    fprintf(stderr, "3MM verification: %s (%zu values, %zu mismatches)\n",
            mismatches ? "FAIL" : "PASS", checked, mismatches);

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_E));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_G));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
