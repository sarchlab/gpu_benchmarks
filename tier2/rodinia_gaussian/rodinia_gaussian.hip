// rodinia_gaussian.hip — Rodinia Gaussian elimination benchmark (HIP, self-contained)
//
// Gaussian elimination with back-substitution to solve Ax=b for an NxN dense matrix.
// Derived from the Rodinia benchmark suite Gaussian workload.
//
// Two GPU kernels:
//   fan1: Compute multipliers m[i][k] = a[i][k] / a[k][k] for pivot row k
//   fan2: Update submatrix: a[i][j] -= m[i][k] * a[k][j], b[i] -= m[i][k] * b[k]
// Back-substitution is performed on the CPU after GPU forward elimination.
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./rodinia_gaussian [--size N]
//
//   --size N         Matrix dimension N (N×N)  (default: 512)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_verify
//   BENCH_PARAM_warmup   (warmup elimination passes, default 3)
//   BENCH_PARAM_iteration    (timed elimination passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (fan1, fan2) carrying that kernel's accumulated GPU event time
// (time_ms) and host time (host_time_ms) across all N-1 pivot steps within
// the pass, then a summary whose total_time_ms is the SUM of all recorded
// GPU event times. GFLOPS is derived downstream from time_ms and params.
// Output (stderr): device info, timing details, verification result

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
// Kernel 1: fan1 — compute multipliers for pivot column t
//
// Each thread handles one row below the pivot:
//   m[(t+1+tid)*Size + t] = a[(t+1+tid)*Size + t] / a[t*Size + t]
// ---------------------------------------------------------------------------
__global__ void fan1(float* __restrict__ m,
                     const float* __restrict__ a,
                     int Size, int t)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Size - 1 - t) {
        int row = t + 1 + tid;
        m[row * Size + t] = a[row * Size + t] / a[t * Size + t];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: fan2 — eliminate pivot column from submatrix below pivot row t
//
// Each (col, row) thread updates one element of the submatrix:
//   a[(t+1+row)*Size + (t+1+col)] -= m[(t+1+row)*Size + t] * a[t*Size + (t+1+col)]
// Thread with col==0 also updates b:
//   b[t+1+row] -= m[(t+1+row)*Size + t] * b[t]
// ---------------------------------------------------------------------------
__global__ void fan2(const float* __restrict__ m,
                     float* __restrict__ a,
                     float* __restrict__ b,
                     int Size, int t)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // relative column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // relative row index
    int remaining = Size - t - 1;

    if (col < remaining && row < remaining) {
        int abs_row = t + 1 + row;
        int abs_col = t + 1 + col;
        a[abs_row * Size + abs_col] -= m[abs_row * Size + t] * a[t * Size + abs_col];
        if (col == 0) {
            b[abs_row] -= m[abs_row * Size + t] * b[t];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU back-substitution: solve upper-triangular system Ux = b
// ---------------------------------------------------------------------------
static void backSubstitution(const float* a, const float* b, float* x, int N)
{
    for (int i = N - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < N; ++j) {
            sum -= (double)a[i * N + j] * (double)x[j];
        }
        x[i] = (float)(sum / (double)a[i * N + i]);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int N     = 512;
    int block_size_param = 256;
    const char* verify = "true";

    // Command-line fallback
    N     = parseIntParam(argc, argv, "--size", N);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size_param = atoi(env_val);
    env_val = getenv("BENCH_PARAM_verify");
    if (env_val) {
        // Validate against the known enum set so the emitted params string is
        // always a canonical literal, never raw env text.
        if      (strcmp(env_val, "true")  == 0) verify = "true";
        else if (strcmp(env_val, "false") == 0) verify = "false";
        else if (strcmp(env_val, "skip")  == 0) verify = "skip";
        else {
            fprintf(stderr, "Invalid BENCH_PARAM_verify (expected true, false "
                            "or skip)\n");
            return 1;
        }
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    size_t bytes_a = (size_t)N * N * sizeof(float);
    size_t bytes_b = (size_t)N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d\n\n", N, N);

    // Host allocations
    float* h_a      = (float*)malloc(bytes_a);
    float* h_b      = (float*)malloc(bytes_b);
    float* h_x      = (float*)malloc(bytes_b);
    float* h_a_orig = (float*)malloc(bytes_a);
    float* h_b_orig = (float*)malloc(bytes_b);

    // Initialize: diagonally dominant matrix for numerical stability (no pivoting needed)
    // A_ii = N (large diagonal), A_ij = random [0, 0.9] for i!=j
    // b_i = random [1.0, 1.9]
    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = (float)(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i * N + i] += (float)N;  // ensure diagonal dominance
    }
    for (int i = 0; i < N; ++i) {
        h_b[i] = (float)(rand() % 10) / 10.0f + 1.0f;
    }

    // Save originals for correctness verification
    memcpy(h_a_orig, h_a, bytes_a);
    memcpy(h_b_orig, h_b, bytes_b);

    // Device allocations
    float *d_a, *d_b, *d_m;
    HIP_CHECK(hipMalloc(&d_a, bytes_a));
    HIP_CHECK(hipMalloc(&d_b, bytes_b));
    HIP_CHECK(hipMalloc(&d_m, bytes_a));  // multiplier matrix (same size as A)

    // Kernel launch configurations
    const int BLOCK1D = 256;
    const int BLOCK2D = 16;

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"verify\":\"%s\"",
             N, block_size_param, verify);

    // Per-kernel accumulators: each elimination pass launches fan1 and fan2
    // once per pivot step; the accumulated GPU/host time of all N-1 steps
    // within one pass becomes that pass's record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_fan1, acc_fan2;

    // One full forward-elimination pass. The elimination destroys A/b in
    // place, so the pristine inputs are re-uploaded at the start of every
    // pass (untimed) — each pass performs identical work.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_fan1);
        bench_lap_reset(&acc_fan2);

        HIP_CHECK(hipMemcpy(d_a, h_a, bytes_a, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b, bytes_b, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_m, 0, bytes_a));

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int t = 0; t < N - 1; ++t) {
            int remaining = N - t - 1;
            // fan1: compute multipliers
            int grid1 = (remaining + BLOCK1D - 1) / BLOCK1D;
            bench_lap_add(&acc_fan1, bench_time_op(&timer, [&]() {
                fan1<<<grid1, BLOCK1D>>>(d_m, d_a, N, t);
            }));
            // fan2: update submatrix
            dim3 block2(BLOCK2D, BLOCK2D);
            dim3 grid2((remaining + BLOCK2D - 1) / BLOCK2D,
                       (remaining + BLOCK2D - 1) / BLOCK2D);
            bench_lap_add(&acc_fan2, bench_time_op(&timer, [&]() {
                fan2<<<grid2, block2>>>(d_m, d_a, d_b, N, t);
            }));
        }
    };

    // Warmup passes (records not emitted)
    for (int w = 0; w < num_warmup; ++w) {
        run_pass();
    }

    // Timed passes — one record per kernel per pass
    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_pass();
        bench_emit_iter("fan1", i, &acc_fan1, params);
        bench_emit_iter("fan2", i, &acc_fan2, params);
        total_gpu_ms += acc_fan1.gpu_ms + acc_fan2.gpu_ms;
        fprintf(stderr, "pass %d: fan1 gpu = %.3f ms, fan2 gpu = %.3f ms\n",
                i, acc_fan1.gpu_ms, acc_fan2.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // Copy final result from device
    HIP_CHECK(hipMemcpy(h_a, d_a, bytes_a, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_b, d_b, bytes_b, hipMemcpyDeviceToHost));

    // CPU back-substitution on the factored system
    backSubstitution(h_a, h_b, h_x, N);

    // Correctness check: compute ||A_orig * x - b_orig|| / ||b_orig||
    double norm_res = 0.0, norm_b = 0.0;
    for (int i = 0; i < N; ++i) {
        double res = -(double)h_b_orig[i];
        for (int j = 0; j < N; ++j) {
            res += (double)h_a_orig[i * N + j] * (double)h_x[j];
        }
        norm_res += res * res;
        norm_b   += (double)h_b_orig[i] * (double)h_b_orig[i];
    }
    double rel_err = sqrt(norm_res / (norm_b + 1e-30));
    bool pass = (rel_err < 1e-4);
    fprintf(stderr, "Verification: rel_err = %.2e — %s\n\n",
            rel_err, pass ? "PASS" : "FAIL");

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_m));
    free(h_a);
    free(h_b);
    free(h_x);
    free(h_a_orig);
    free(h_b_orig);

    return pass ? 0 : 1;
}
