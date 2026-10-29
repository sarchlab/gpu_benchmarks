// rodinia_lud.hip — Rodinia LUD benchmark (HIP, self-contained)
//
// Blocked LU decomposition of a dense NxN matrix (no pivoting).
// Derived from the Rodinia LUD benchmark suite.
//
// Three GPU kernels:
//   lud_diagonal   — in-place LU factor of the diagonal 16×16 block
//   lud_perimeter  — forward/back-solve for row/column perimeter blocks
//   lud_internal   — Schur-complement update for interior blocks
//
// Default: N=512, BSIZE=16  (N must be divisible by BSIZE)
//
// Usage:
//   ./rodinia_lud [--size N]
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_verify
//
// Output (stdout): one {"type":"kernel",...} JSON line per
// kernel (lud_diagonal, lud_perimeter, lud_internal) carrying that kernel's
// accumulated GPU event time across all block steps in the pass plus the
// host wall-clock time, then a summary whose total_time_ms is the SUM of
// all recorded GPU event times. GFLOPS is derived downstream from time_ms
// and params.
// Output (stderr): device info, per-pass timing, verification result

#include "../../../common_hip/bench_timing.h"
#include "lud_residual.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Constants and error checking
// ---------------------------------------------------------------------------

#define BSIZE 16

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Factor the diagonal block at matrix position (offset*BSIZE, offset*BSIZE).
// Launch: <<<dim3(1), dim3(BSIZE,BSIZE)>>>
__global__ void lud_diagonal(float *a, int n, int offset)
{
    __shared__ float s[BSIZE][BSIZE];
    int tx = threadIdx.x, ty = threadIdx.y;

    s[ty][tx] = a[(offset * BSIZE + ty) * n + (offset * BSIZE + tx)];
    __syncthreads();

    for (int k = 0; k < BSIZE - 1; k++) {
        // Normalise column k below pivot  → L[i][k] = a[i][k] / a[k][k]
        if (ty > k && tx == k)
            s[ty][k] /= s[k][k];
        __syncthreads();
        // Schur complement on trailing sub-block
        if (ty > k && tx > k)
            s[ty][tx] -= s[ty][k] * s[k][tx];
        __syncthreads();
    }

    a[(offset * BSIZE + ty) * n + (offset * BSIZE + tx)] = s[ty][tx];
}

// Update perimeter blocks adjacent to the diagonal at step `offset`.
// First half of blocks (blockIdx.x < half): row panel (right of diagonal)
//   → forward-solve  L_kk * U_kj = A_kj
// Second half (blockIdx.x >= half):          column panel (below diagonal)
//   → back-solve     L_ik * U_kk = A_ik
// Launch: <<<dim3(2*half), dim3(BSIZE,BSIZE)>>>  where half = num_blocks-offset-1
__global__ void lud_perimeter(float *a, int n, int offset)
{
    __shared__ float dia [BSIZE][BSIZE];
    __shared__ float peri[BSIZE][BSIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int half   = gridDim.x / 2;
    bool is_row = (blockIdx.x < half);
    int  idx    = is_row ? blockIdx.x : (blockIdx.x - half);
    int  blk    = offset + idx + 1;   // target block index

    // All threads in this block load the same diagonal block
    dia[ty][tx] = a[(offset * BSIZE + ty) * n + (offset * BSIZE + tx)];

    if (is_row)
        peri[ty][tx] = a[(offset * BSIZE + ty) * n + (blk * BSIZE + tx)];
    else
        peri[ty][tx] = a[(blk   * BSIZE + ty) * n + (offset * BSIZE + tx)];
    __syncthreads();

    // is_row is uniform across the entire thread block (depends only on blockIdx.x),
    // so __syncthreads() inside these branches is valid.
    if (is_row) {
        // Forward substitution: unit-lower-triangular L * peri = peri
        for (int k = 0; k < BSIZE - 1; k++) {
            if (ty > k)
                peri[ty][tx] -= dia[ty][k] * peri[k][tx];
            __syncthreads();
        }
        a[(offset * BSIZE + ty) * n + (blk * BSIZE + tx)] = peri[ty][tx];
    } else {
        // Back substitution: peri * U = peri  → find L column
        for (int k = 0; k < BSIZE; k++) {
            if (tx == k)
                peri[ty][k] /= dia[k][k];
            __syncthreads();
            if (tx > k)
                peri[ty][tx] -= peri[ty][k] * dia[k][tx];
            __syncthreads();
        }
        a[(blk * BSIZE + ty) * n + (offset * BSIZE + tx)] = peri[ty][tx];
    }
}

// Schur-complement update for interior block (row_blk, col_blk).
// A[row_blk][col_blk] -= L[row_blk][offset] * U[offset][col_blk]
// Launch: <<<dim3(intern, intern), dim3(BSIZE,BSIZE)>>>  where intern = num_blocks-offset-1
__global__ void lud_internal(float *a, int n, int offset)
{
    __shared__ float peri_row[BSIZE][BSIZE];  // U[offset, col_blk]
    __shared__ float peri_col[BSIZE][BSIZE];  // L[row_blk, offset]

    int tx = threadIdx.x, ty = threadIdx.y;
    int col_blk = offset + blockIdx.x + 1;
    int row_blk = offset + blockIdx.y + 1;

    peri_row[ty][tx] = a[(offset  * BSIZE + ty) * n + (col_blk * BSIZE + tx)];
    peri_col[ty][tx] = a[(row_blk * BSIZE + ty) * n + (offset  * BSIZE + tx)];
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < BSIZE; k++)
        sum += peri_col[ty][k] * peri_row[k][tx];
    a[(row_blk * BSIZE + ty) * n + (col_blk * BSIZE + tx)] -= sum;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Diagonally-dominant random matrix → LU without pivoting is numerically stable
static void init_matrix(float *a, int n)
{
    srand(42);
    for (int i = 0; i < n; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (float)(rand() % 10 + 1) * 0.1f;
            if (i != j) row_sum += fabsf(a[i * n + j]);
        }
        a[i * n + i] = row_sum + 1.0f;  // ensure diagonal dominance
    }
}

// Verify: reconstruct A = L*U from the in-place result and compare with original.
// In-place storage: lower-triangular (i>j) → L (unit diagonal implicit)
//                   upper-triangular (i<=j) → U
static bool verify(const float *lu, const float *a_orig, int n)
{
    const double rel = lud_reference::relative_residual(lu, a_orig, n);
    fprintf(stderr, "Verification: ||A - LU||_F / ||A||_F = %.6e  %s\n",
            rel, (rel < 1e-4) ? "(PASS)" : "(FAIL)");
    return rel < 1e-4;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int N     = 512;

    // Command-line fallback
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--size")       == 0) N     = atoi(argv[i + 1]);
    }

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);

    // Debug knob (not a sweep parameter): CPU verification runs by default;
    // "false"/"0"/"skip" skip it.
    bool do_verify = true;
    env_val = getenv("BENCH_PARAM_verify");
    if (env_val && (strcmp(env_val, "false") == 0 ||
                    strcmp(env_val, "0")     == 0 ||
                    strcmp(env_val, "skip")  == 0)) {
        do_verify = false;
    }
    if (N % BSIZE != 0) {
        fprintf(stderr, "Error: N=%d must be divisible by BSIZE=%d\n", N, BSIZE);
        return 1;
    }

    int    num_blocks = N / BSIZE;
    size_t bytes      = (size_t)N * N * sizeof(float);

    // Device info
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "Device: %s\n", prop.name);
    fprintf(stderr, "Matrix: %dx%d  BSIZE=%d  num_blocks=%d\n\n",
            N, N, BSIZE, num_blocks);

    // Host allocations
    float *h_a      = (float*)malloc(bytes);
    float *h_a_orig = (float*)malloc(bytes);
    float *h_lu     = (float*)malloc(bytes);

    init_matrix(h_a, N);
    memcpy(h_a_orig, h_a, bytes);

    // Device allocation
    float *d_a;
    HIP_CHECK(hipMalloc(&d_a, bytes));

    dim3 block(BSIZE, BSIZE);

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"verification\":\"%s\"",
             N, do_verify ? "full_lu_residual_v1" : "disabled");

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_diagonal, acc_perimeter, acc_internal;

    // One full LUD pass. The factorization is IN PLACE, so the matrix MUST be
    // re-uploaded from the pristine host copy at the start of every pass
    // (untimed) — otherwise later passes would factor an already-factored
    // matrix. Only the kernel launches are inside the timed brackets.
    auto run_lud_pass = [&]() {
        bench_lap_reset(&acc_diagonal);
        bench_lap_reset(&acc_perimeter);
        bench_lap_reset(&acc_internal);

        HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int k = 0; k < num_blocks; k++) {
            bench_lap_add(&acc_diagonal, bench_time_op(&timer, [&]() {
                lud_diagonal<<<dim3(1), block>>>(d_a, N, k);
            }));
            int peri   = 2 * (num_blocks - k - 1);
            int intern =     num_blocks - k - 1;
            if (peri > 0) {
                bench_lap_add(&acc_perimeter, bench_time_op(&timer, [&]() {
                    lud_perimeter<<<dim3(peri), block>>>(d_a, N, k);
                }));
                bench_lap_add(&acc_internal, bench_time_op(&timer, [&]() {
                    lud_internal<<<dim3(intern,intern), block>>>(d_a, N, k);
                }));
            }
        }
    };

    // One measured factorization; verification below checks this same result.
    double total_gpu_ms = 0.0;
    run_lud_pass();
    bench_emit_iter("lud_diagonal",  0, &acc_diagonal,  params);
    bench_emit_iter("lud_perimeter", 0, &acc_perimeter, params);
    bench_emit_iter("lud_internal",  0, &acc_internal,  params);
    total_gpu_ms += acc_diagonal.gpu_ms + acc_perimeter.gpu_ms
                  + acc_internal.gpu_ms;
    fprintf(stderr, "diagonal gpu = %.3f ms, perimeter gpu = %.3f ms, "
                    "internal gpu = %.3f ms\n",
            acc_diagonal.gpu_ms, acc_perimeter.gpu_ms, acc_internal.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify the actual measured result after all event/host launch timing.
    // Preserve the existing full Frobenius residual and optional debug bypass;
    // no second factorization or entry warmer is needed for verification.
    bool verify_ok = true;
    if (do_verify) {
        HIP_CHECK(hipMemcpy(h_lu, d_a, bytes, hipMemcpyDeviceToHost));
        verify_ok = verify(h_lu, h_a_orig, N);
    }

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    free(h_a);
    free(h_a_orig);
    free(h_lu);

    return verify_ok ? 0 : 1;
}
