// polybench_jacobi2d.hip — PolyBench Jacobi 2D stencil benchmark (HIP, self-contained)
//
// Computes TSTEPS iterations of 2D Jacobi stencil on an N×N grid:
//   B[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] + A[i][j]) / 5.0
//   then swap A and B (double-buffer)
//
// Only interior points (i=1..N-2, j=1..N-2) are updated; boundaries remain 0.
//
// Default: N=1024, TSTEPS=50
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./polybench_jacobi2d [--n N] [--tsteps T] [--trials K]
//
//   --n N        Grid size N×N           (default: 1024)
//   --tsteps T   Number of time steps    (default: 50)
//   --trials K   Legacy trial count (echoed in params; the measured pass
//                count is BENCH_PARAM_num_iters)
//
// Environment variables (override argv):
//   BENCH_PARAM_n, BENCH_PARAM_tsteps, BENCH_PARAM_trials, BENCH_PARAM_block_size
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass (one pass = TSTEPS kernel launches), one
// {"type":"kernel",...} JSON line carrying the pass's accumulated GPU event
// time (time_ms) and host wall-clock time (host_time_ms), then a summary
// whose total_time_ms is the SUM of all recorded GPU event times.
// Bandwidth (GB/s) is derived downstream from time_ms and params.
// Output (stderr): device info, correctness check, per-pass timing

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Inlined utilities
// ---------------------------------------------------------------------------

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
// Kernel: one step of 2D Jacobi stencil
// ---------------------------------------------------------------------------
#define BLOCK_DIM 16

__global__ void jacobi2d_kernel(const float* __restrict__ A,
                                 float* __restrict__ B,
                                 int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // interior row
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  // interior col
    if (i >= N-1 || j >= N-1) return;
    B[i*N + j] = (A[(i-1)*N + j] + A[(i+1)*N + j] +
                  A[i*N + (j-1)] + A[i*N + (j+1)] +
                  A[i*N + j]) * 0.2f;
}

// ---------------------------------------------------------------------------
// CPU reference for correctness check
// ---------------------------------------------------------------------------
static void cpu_jacobi2d(const float* A, float* B, int N) {
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            B[i*N + j] = (A[(i-1)*N + j] + A[(i+1)*N + j] +
                          A[i*N + (j-1)] + A[i*N + (j+1)] +
                          A[i*N + j]) * 0.2f;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N      = parseIntParam(argc, argv, "--n",      1024);
    int TSTEPS = parseIntParam(argc, argv, "--tsteps", 50);
    int trials = parseIntParam(argc, argv, "--trials", 3);
    int block_size_param = parseIntParam(argc, argv, "--block_size", 256);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_tsteps");
    if (env_val) TSTEPS = atoi(env_val);
    env_val = getenv("BENCH_PARAM_trials");
    if (env_val) trials = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size_param = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    size_t bytes = (size_t)N * N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d  |  TSTEPS: %d  |  %d warmup + %d timed passes\n\n",
            N, N, TSTEPS, num_warmup, num_iters);

    // Host allocations
    std::vector<float> h_A((size_t)N * N, 0.0f);

    // Initialize with random floats (interior only; boundaries stay 0)
    srand(42);
    for (int i = 1; i < N-1; ++i)
        for (int j = 1; j < N-1; ++j)
            h_A[i*N + j] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));

    // Copy A to device (B starts zeroed)
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B, 0, bytes));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N-2 + BLOCK_DIM - 1) / BLOCK_DIM,
              (N-2 + BLOCK_DIM - 1) / BLOCK_DIM);

    char params[160];
    snprintf(params, sizeof(params),
             "\"n\":%d,\"tsteps\":%d,\"trials\":%d,\"block_size\":%d",
             N, TSTEPS, trials, block_size_param);

    // Per-pass accumulator: one pass = TSTEPS jacobi2d_kernel launches,
    // each launch individually bracketed.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_jacobi;

    // One full Jacobi pass. The two grids ping-pong across the TSTEPS
    // internal steps; the initial state is restored at the start of every
    // pass (untimed) — the same reset the original code performed. The step
    // count is fixed (no convergence test), and the stencil timing is
    // value-independent, so each pass measures identical work.
    auto run_jacobi_pass = [&]() {
        bench_lap_reset(&acc_jacobi);

        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_B, 0, bytes));

        float *src = d_A, *dst = d_B;
        for (int t = 0; t < TSTEPS; ++t) {
            bench_lap_add(&acc_jacobi, bench_time_op(&timer, [&]() {
                jacobi2d_kernel<<<grid, block>>>(src, dst, N);
            }));
            float* tmp = src; src = dst; dst = tmp;
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_jacobi_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_jacobi_pass();
        bench_emit_iter("jacobi2d_kernel", i, &acc_jacobi, params);
        total_gpu_ms += acc_jacobi.gpu_ms;
        fprintf(stderr, "pass %d: gpu = %.3f ms (%d steps)\n",
                i, acc_jacobi.gpu_ms, TSTEPS);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // Correctness check: run CPU for TSTEPS on small 64×64 grid
    {
        int CN = 64;
        std::vector<float> cA((size_t)CN * CN, 0.0f);
        std::vector<float> cB((size_t)CN * CN, 0.0f);
        std::vector<float> cA_gpu_init((size_t)CN * CN, 0.0f);

        srand(42);
        for (int i = 1; i < CN-1; ++i)
            for (int j = 1; j < CN-1; ++j)
                cA[i*CN + j] = (float)(rand() % 100) / 10.0f;
        cA_gpu_init = cA;

        // Run CPU TSTEPS
        for (int t = 0; t < TSTEPS; ++t) {
            cpu_jacobi2d(cA.data(), cB.data(), CN);
            cA.swap(cB);
        }
        // cA now holds CPU result

        // Run GPU on CN×CN grid
        size_t cbytes = (size_t)CN * CN * sizeof(float);
        float *cd_A, *cd_B;
        HIP_CHECK(hipMalloc(&cd_A, cbytes));
        HIP_CHECK(hipMalloc(&cd_B, cbytes));
        HIP_CHECK(hipMemcpy(cd_A, cA_gpu_init.data(), cbytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(cd_B, 0, cbytes));

        dim3 cblock(BLOCK_DIM, BLOCK_DIM);
        dim3 cgrid((CN-2 + BLOCK_DIM - 1) / BLOCK_DIM,
                   (CN-2 + BLOCK_DIM - 1) / BLOCK_DIM);

        float *csrc = cd_A, *cdst = cd_B;
        for (int t = 0; t < TSTEPS; ++t) {
            jacobi2d_kernel<<<cgrid, cblock>>>(csrc, cdst, CN);
            float* tmp = csrc; csrc = cdst; cdst = tmp;
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> gpu_result((size_t)CN * CN);
        float *cfinal = (TSTEPS % 2 == 0) ? cd_A : cd_B;
        HIP_CHECK(hipMemcpy(gpu_result.data(), cfinal, cbytes, hipMemcpyDeviceToHost));

        bool pass = true;
        float max_rel = 0.0f;
        for (int i = 1; i < CN-1 && pass; ++i) {
            for (int j = 1; j < CN-1; ++j) {
                float ref = cA[i*CN + j];
                float gpu = gpu_result[i*CN + j];
                float rel = fabsf(gpu - ref) / (fabsf(ref) + 1e-6f);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-3f) {
                    fprintf(stderr, "MISMATCH [%d][%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                            i, j, gpu, ref, rel);
                    pass = false;
                    break;
                }
            }
        }
        fprintf(stderr, "Correctness check (CPU vs GPU, %d×%d, %d steps): %s (max_rel=%.2e)\n\n",
                CN, CN, TSTEPS, pass ? "PASS" : "FAIL", max_rel);

        HIP_CHECK(hipFree(cd_A));
        HIP_CHECK(hipFree(cd_B));
    }

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));

    return 0;
}
