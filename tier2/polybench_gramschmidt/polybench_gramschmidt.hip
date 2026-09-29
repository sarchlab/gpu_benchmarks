// polybench_gramschmidt.hip — PolyBench Gram-Schmidt orthogonalization (HIP, self-contained)
//
// Computes the QR factorization via Gram-Schmidt orthogonalization:
//   A = Q * R
// where A is M×N, Q is M×N (orthonormal columns), R is N×N (upper triangular).
// Derived from the PolyBench/GPU benchmark suite.
//
// GPU kernels (executed in a loop over k = 0 .. N-1):
//   gram_norm_zero    : reset the scalar norm accumulator (1 thread)
//   gram_norm         : atomic-reduction of ||A[:,k]||^2 into nrm_buf
//   gram_norm_finish  : nrm_buf = sqrt(nrm_buf); R[k,k] = nrm_buf  (1 thread)
//   gram_normalize    : Q[:,k] = A[:,k] / nrm_buf                   (M threads)
//   gram_project      : for j>k: R[k,j]=dot(Q[:,k],A[:,j]); A[:,j]-=R[k,j]*Q[:,k]
//                       one thread per column j > k
//
// Default: M = 512, N = 64
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./polybench_gramschmidt [--m M] [--n N]
//
//   --m M       Number of rows                          (default: 512)
//   --n N       Number of columns                       (default: 64)
//
// Environment variables (override argv):
//   BENCH_PARAM_m, BENCH_PARAM_n
//
// Output (stdout): one {"type":"kernel",...} JSON line per kernel
// (gram_norm_zero, gram_norm, gram_norm_finish, gram_normalize, gram_project) carrying that
// kernel's GPU event time (time_ms) and host wall-clock time (host_time_ms)
// accumulated over all k-columns of the single QR factorization, then a
// summary whose total_time_ms is the SUM of all recorded GPU event times.
// GB/s is derived downstream from time_ms and params.
// Output (stderr): device info, timing details, orthogonality check

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
// Kernel: reset the norm accumulator before each column's reduction
// ---------------------------------------------------------------------------
__global__ void gram_norm_zero(float* nrm_buf)
{
    *nrm_buf = 0.0f;
}

// ---------------------------------------------------------------------------
// Kernel: accumulate ||A[:,k]||^2 via atomic add
// ---------------------------------------------------------------------------
__global__ void gram_norm(const float* __restrict__ A,
                           float* __restrict__ nrm_buf,
                           int M, int N, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) return;
    float val = A[i * N + k];
    atomicAdd(nrm_buf, val * val);
}

// ---------------------------------------------------------------------------
// Kernel: finalize norm — single thread computes sqrt, stores in R[k,k]
// ---------------------------------------------------------------------------
__global__ void gram_norm_finish(float* __restrict__ nrm_buf,
                                  float* __restrict__ R,
                                  int N, int k)
{
    float nrm = sqrtf(*nrm_buf);
    R[k * N + k] = nrm;
    *nrm_buf = nrm;  // overwrite so normalize kernel can read it
}

// ---------------------------------------------------------------------------
// Kernel: Q[:,k] = A[:,k] / nrm_buf[0]
// ---------------------------------------------------------------------------
__global__ void gram_normalize(const float* __restrict__ A,
                                float* __restrict__ Q,
                                const float* __restrict__ nrm_buf,
                                int M, int N, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) return;
    Q[i * N + k] = A[i * N + k] / nrm_buf[0];
}

// ---------------------------------------------------------------------------
// Kernel: update columns j = k+1..N-1
//   dot   = Q[:,k] · A[:,j]
//   R[k,j] = dot
//   A[:,j] -= dot * Q[:,k]
// One thread per column j > k.
// ---------------------------------------------------------------------------
__global__ void gram_project(float* __restrict__ A,
                              const float* __restrict__ Q,
                              float* __restrict__ R,
                              int M, int N, int k)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x + k + 1;
    if (j >= N) return;

    float dot = 0.0f;
    for (int i = 0; i < M; i++)
        dot += Q[i * N + k] * A[i * N + j];
    R[k * N + j] = dot;
    for (int i = 0; i < M; i++)
        A[i * N + j] -= dot * Q[i * N + k];
}

// ---------------------------------------------------------------------------
// CPU orthogonality reference check: verify Q^T * Q ≈ I
// Checks every column pair of the actual measured Q.
// ---------------------------------------------------------------------------
static bool check_orthogonality(const float* Q, int M, int N)
{
    size_t errors = 0;
    for (int ci = 0; ci < N; ci++) {
        for (int cj = 0; cj < N; cj++) {
            float dot = 0.0f;
            for (int i = 0; i < M; i++)
                dot += Q[i * N + ci] * Q[i * N + cj];
            float expected = (ci == cj) ? 1.0f : 0.0f;
            if (!std::isfinite(dot) || !std::isfinite(expected) ||
                fabsf(dot - expected) > 0.01f) {
                if (errors < 10)
                    fprintf(stderr,
                            "ORTH FAIL: dot(Q[:,%d], Q[:,%d]) = %.6f  (expected %.1f)\n",
                            ci, cj, dot, expected);
                ++errors;
            }
        }
    }
    fprintf(stderr, "Orthogonality: %zu failing pairs out of %zu\n",
            errors, (size_t)N * N);
    return errors == 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    int M      = parseIntParam(argc, argv, "--m", 512);
    int N      = parseIntParam(argc, argv, "--n", 64);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_m");
    if (env_val) M = atoi(env_val);
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) N = atoi(env_val);

    size_t bytes_A = (size_t)M * N * sizeof(float);
    size_t bytes_Q = (size_t)M * N * sizeof(float);
    size_t bytes_R = (size_t)N * N * sizeof(float);
    size_t bytes_1 = sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix: %d×%d\n\n", M, N);

    // Host allocations
    float* h_A_orig = (float*)malloc(bytes_A);
    float* h_Q      = (float*)malloc(bytes_Q);

    if (!h_A_orig || !h_Q) {
        fprintf(stderr, "Error: host malloc failed\n");
        return 1;
    }

    // Initialize A with random floats in [0,1) to ensure full column rank
    srand(42);
    for (int i = 0; i < M * N; ++i)
        h_A_orig[i] = (float)rand() / (float)RAND_MAX;

    // Device allocations
    float *d_A, *d_Q, *d_R, *d_nrm_buf;
    HIP_CHECK(hipMalloc(&d_A,       bytes_A));
    HIP_CHECK(hipMalloc(&d_Q,       bytes_Q));
    HIP_CHECK(hipMalloc(&d_R,       bytes_R));
    HIP_CHECK(hipMalloc(&d_nrm_buf, bytes_1));

    const int block_size = 256;
    const int grid_M = (M + block_size - 1) / block_size;

    char params[160];
    snprintf(params, sizeof(params),
             "\"m\":%d,\"n\":%d,\"q_orthogonality\":\"all_columns_v1\"",
             M, N);

    // Per-kernel accumulators: the single run = one full QR factorization; each
    // kernel's GPU/host time is accumulated over all k-columns, giving one
    // record per kernel. Repetition across processes is the harness's job
    // (gpubench --reps) — the operation runs exactly once here.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_zero, acc_norm, acc_norm_finish, acc_normalize, acc_project;

    // The single recorded run. Gram-Schmidt orthogonalizes A in place, so A is
    // restored from the pristine host copy (and Q/R zeroed) before the
    // factorization. Initial setup stays outside timed brackets; each column's
    // required accumulator reset is a measured kernel inside the algorithm.
    auto run_iter = [&]() {
        bench_lap_reset(&acc_zero);
        bench_lap_reset(&acc_norm);
        bench_lap_reset(&acc_norm_finish);
        bench_lap_reset(&acc_normalize);
        bench_lap_reset(&acc_project);

        // Restore A from original, zero Q and R (outside timing)
        HIP_CHECK(hipMemcpy(d_A, h_A_orig, bytes_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_Q, 0, bytes_Q));
        HIP_CHECK(hipMemset(d_R, 0, bytes_R));
        HIP_CHECK(hipDeviceSynchronize());

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the H2D re-init so the first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int k = 0; k < N; ++k) {
            // Step 1: reset and then accumulate ||A[:,k]||^2. A separate
            // launch provides the global ordering required by the reduction.
            bench_lap_add(&acc_zero, bench_time_op(&timer, [&]() {
                gram_norm_zero<<<1, 1>>>(d_nrm_buf);
            }));
            bench_lap_add(&acc_norm, bench_time_op(&timer, [&]() {
                gram_norm<<<grid_M, block_size>>>(d_A, d_nrm_buf, M, N, k);
            }));

            // Step 2: finalize norm → R[k,k]
            bench_lap_add(&acc_norm_finish, bench_time_op(&timer, [&]() {
                gram_norm_finish<<<1, 1>>>(d_nrm_buf, d_R, N, k);
            }));

            // Step 3: normalize column k
            bench_lap_add(&acc_normalize, bench_time_op(&timer, [&]() {
                gram_normalize<<<grid_M, block_size>>>(d_A, d_Q, d_nrm_buf, M, N, k);
            }));

            // Step 4: project remaining columns (if any)
            if (k + 1 < N) {
                int remaining   = N - k - 1;
                int grid_proj   = (remaining + block_size - 1) / block_size;
                bench_lap_add(&acc_project, bench_time_op(&timer, [&]() {
                    gram_project<<<grid_proj, block_size>>>(d_A, d_Q, d_R, M, N, k);
                }));
            }
        }
    };

    run_iter();

    double total_gpu_ms = 0.0;
    bench_emit_iter("gram_norm_zero",   0, &acc_zero,        params);
    bench_emit_iter("gram_norm",        0, &acc_norm,        params);
    bench_emit_iter("gram_norm_finish", 0, &acc_norm_finish, params);
    bench_emit_iter("gram_normalize",   0, &acc_normalize,   params);
    bench_emit_iter("gram_project",     0, &acc_project,     params);
    total_gpu_ms += acc_zero.gpu_ms + acc_norm.gpu_ms + acc_norm_finish.gpu_ms +
                    acc_normalize.gpu_ms + acc_project.gpu_ms;
    fprintf(stderr,
            "zero gpu = %.3f ms, norm gpu = %.3f ms, finish gpu = %.3f ms, "
            "normalize gpu = %.3f ms, project gpu = %.3f ms\n",
            acc_zero.gpu_ms, acc_norm.gpu_ms, acc_norm_finish.gpu_ms,
            acc_normalize.gpu_ms, acc_project.gpu_ms);

    bench_timer_destroy(&timer);

    // Copy Q back for orthogonality check
    HIP_CHECK(hipMemcpy(h_Q, d_Q, bytes_Q, hipMemcpyDeviceToHost));

    // Orthogonality check
    bool orth_ok = check_orthogonality(h_Q, M, N);
    fprintf(stderr, "Orthogonality check (all %d columns): %s\n\n",
            N, orth_ok ? "PASS" : "FAIL");

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_R));
    HIP_CHECK(hipFree(d_nrm_buf));
    free(h_A_orig);
    free(h_Q);

    return orth_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
