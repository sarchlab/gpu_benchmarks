// npb_cg.hip — NPB Conjugate Gradient benchmark (HIP/ROCm)
//
// Sparse iterative CG solver: SpMV (CSR format) + dot products + AXPY
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer.
//
// Environment variables:
//   BENCH_PARAM_matrix_size, BENCH_PARAM_block_size,
//   BENCH_PARAM_max_cg_iterations, BENCH_PARAM_nonzeros_per_row
//   BENCH_PARAM_warmup   (warmup CG solves, default 3)
//   BENCH_PARAM_iteration    (timed CG solves, each recorded, default 7)
//
// Output (stdout): per timed CG solve, one {"type":"kernel",...} JSON line per
// kernel (spmv_kernel, dot_product_kernel, axpy_kernel, scale_kernel) carrying
// that kernel's accumulated GPU event time and host time within the solve,
// then a summary whose total_time_ms is the SUM of all recorded GPU event
// times. Bandwidth metrics are derived downstream from time_ms and params.
// Output (stderr): human-readable

#include "../../common/bench_timing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Kernel: Sparse Matrix-Vector Multiply (CSR format)
// ---------------------------------------------------------------------------

__global__ void spmv_kernel(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    for (int j = start; j < end; ++j) {
        sum += values[j] * x[col_idx[j]];
    }
    y[row] = sum;
}

// ---------------------------------------------------------------------------
// Kernel: Dot product using block-level reduction
// ---------------------------------------------------------------------------

__global__ void dot_product_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ partial,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) val = a[idx] * b[idx];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Kernel: AXPY  y = alpha * x + y
// ---------------------------------------------------------------------------

__global__ void axpy_kernel(
    float        alpha,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// ---------------------------------------------------------------------------
// Kernel: Scale  y = alpha * x
// ---------------------------------------------------------------------------

__global__ void scale_kernel(
    float        alpha,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = alpha * x[idx];
    }
}

// ---------------------------------------------------------------------------
// Host helper: finish dot product
// ---------------------------------------------------------------------------

static float finish_dot(float* d_partial, int numBlocks) {
    std::vector<float> h_partial(numBlocks);
    HIP_CHECK(hipMemcpy(h_partial.data(), d_partial,
                          numBlocks * sizeof(float), hipMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) sum += h_partial[i];
    return sum;
}

// ---------------------------------------------------------------------------
// Simple PRNG
// ---------------------------------------------------------------------------

static inline unsigned int lcg_rand(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N              = 50000;
    int block_size     = 256;
    int max_cg_iters   = 25;
    int nonzeros_per_row = 7;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_matrix_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_max_cg_iterations");
    if (env_val) max_cg_iters = atoi(env_val);
    env_val = getenv("BENCH_PARAM_nonzeros_per_row");
    if (env_val) nonzeros_per_row = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--size") == 0) N = atoi(argv[i+1]);
    }

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "NPB CG  |  Matrix: %d x %d  |  NNZ/row: %d  |  "
            "CG iters: %d  |  Solves: %d warmup + %d timed\n\n",
            N, N, nonzeros_per_row, max_cg_iters, num_warmup, num_iters);

    // Generate synthetic sparse matrix in CSR format
    int nnz_per_row = nonzeros_per_row;
    long long total_nnz = (long long)N * nnz_per_row;

    std::vector<int>   h_row_ptr(N + 1);
    std::vector<int>   h_col_idx(total_nnz);
    std::vector<float> h_values(total_nnz);
    std::vector<float> h_x(N);
    std::vector<float> h_b(N);

    unsigned int seed = 42;

    for (int i = 0; i < N; ++i) {
        h_row_ptr[i] = i * nnz_per_row;
    }
    h_row_ptr[N] = (int)total_nnz;

    for (int i = 0; i < N; ++i) {
        int base = i * nnz_per_row;
        h_col_idx[base] = i;
        h_values[base]  = (float)(nnz_per_row + 1);
        for (int k = 1; k < nnz_per_row; ++k) {
            int col;
            do { col = lcg_rand(&seed) % N; } while (col == i);
            h_col_idx[base + k] = col;
            h_values[base + k]  = -1.0f;
        }
    }

    for (int i = 0; i < N; ++i) {
        h_x[i] = 0.0f;
        h_b[i] = 1.0f;
    }

    // Device allocations
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_b;
    float *d_r, *d_p, *d_Ap;
    float *d_partial;

    size_t ptr_bytes = (N + 1) * sizeof(int);
    size_t idx_bytes = total_nnz * sizeof(int);
    size_t val_bytes = total_nnz * sizeof(float);
    size_t vec_bytes = N * sizeof(float);

    int gridSize  = (N + block_size - 1) / block_size;
    int numBlocks = gridSize;

    HIP_CHECK(hipMalloc(&d_row_ptr, ptr_bytes));
    HIP_CHECK(hipMalloc(&d_col_idx, idx_bytes));
    HIP_CHECK(hipMalloc(&d_values,  val_bytes));
    HIP_CHECK(hipMalloc(&d_x,       vec_bytes));
    HIP_CHECK(hipMalloc(&d_b,       vec_bytes));
    HIP_CHECK(hipMalloc(&d_r,       vec_bytes));
    HIP_CHECK(hipMalloc(&d_p,       vec_bytes));
    HIP_CHECK(hipMalloc(&d_Ap,      vec_bytes));
    HIP_CHECK(hipMalloc(&d_partial, numBlocks * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr.data(), ptr_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, h_col_idx.data(), idx_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values,  h_values.data(),  val_bytes, hipMemcpyHostToDevice));

    int smem = block_size * sizeof(float);

    char params[192];
    snprintf(params, sizeof(params),
             "\"matrix_size\":%d,\"block_size\":%d,"
             "\"max_cg_iterations\":%d,\"nonzeros_per_row\":%d",
             N, block_size, max_cg_iters, nonzeros_per_row);

    // Per-kernel accumulators: one CG solve launches each kernel many times;
    // the accumulated GPU/host time within one solve becomes that solve's
    // record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_spmv, acc_dot, acc_axpy, acc_scale;

    // One full CG solve. x, b, r, p are re-initialized at the start of every
    // solve (untimed), so each solve runs exactly max_cg_iters identical
    // iterations regardless of state left by earlier solves.
    auto run_cg = [&]() {
        bench_lap_reset(&acc_spmv);
        bench_lap_reset(&acc_dot);
        bench_lap_reset(&acc_axpy);
        bench_lap_reset(&acc_scale);

        HIP_CHECK(hipMemcpy(d_x, h_x.data(), vec_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b.data(), vec_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_r, d_b, vec_bytes, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_p, d_r, vec_bytes, hipMemcpyDeviceToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_dot, bench_time_op(&timer, [&]() {
            dot_product_kernel<<<numBlocks, block_size, smem>>>(d_r, d_r, d_partial, N);
        }));
        float rr = finish_dot(d_partial, numBlocks);

        for (int iter = 0; iter < max_cg_iters; ++iter) {
            bench_lap_add(&acc_spmv, bench_time_op(&timer, [&]() {
                spmv_kernel<<<gridSize, block_size>>>(d_row_ptr, d_col_idx, d_values, d_p, d_Ap, N);
            }));

            bench_lap_add(&acc_dot, bench_time_op(&timer, [&]() {
                dot_product_kernel<<<numBlocks, block_size, smem>>>(d_p, d_Ap, d_partial, N);
            }));
            float pAp = finish_dot(d_partial, numBlocks);
            float alpha = rr / fmaxf(pAp, 1e-20f);

            bench_lap_add(&acc_axpy, bench_time_op(&timer, [&]() {
                axpy_kernel<<<gridSize, block_size>>>(alpha, d_p, d_x, N);
            }));
            bench_lap_add(&acc_axpy, bench_time_op(&timer, [&]() {
                axpy_kernel<<<gridSize, block_size>>>(-alpha, d_Ap, d_r, N);
            }));

            bench_lap_add(&acc_dot, bench_time_op(&timer, [&]() {
                dot_product_kernel<<<numBlocks, block_size, smem>>>(d_r, d_r, d_partial, N);
            }));
            float rr_new = finish_dot(d_partial, numBlocks);

            float beta = rr_new / fmaxf(rr, 1e-20f);
            bench_lap_add(&acc_scale, bench_time_op(&timer, [&]() {
                scale_kernel<<<gridSize, block_size>>>(beta, d_p, d_p, N);
            }));

            bench_lap_add(&acc_axpy, bench_time_op(&timer, [&]() {
                axpy_kernel<<<gridSize, block_size>>>(1.0f, d_r, d_p, N);
            }));

            rr = rr_new;
        }
    };

    // Warmup
    for (int w = 0; w < num_warmup; ++w) {
        run_cg();
    }

    // Timed solves — one record per kernel per solve
    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_cg();
        bench_emit_iter("spmv_kernel",        i, &acc_spmv,  params);
        bench_emit_iter("dot_product_kernel", i, &acc_dot,   params);
        bench_emit_iter("axpy_kernel",        i, &acc_axpy,  params);
        bench_emit_iter("scale_kernel",       i, &acc_scale, params);
        total_gpu_ms += acc_spmv.gpu_ms + acc_dot.gpu_ms +
                        acc_axpy.gpu_ms + acc_scale.gpu_ms;
        fprintf(stderr, "solve %d: spmv gpu = %.3f ms, dot gpu = %.3f ms, "
                "axpy gpu = %.3f ms, scale gpu = %.3f ms\n",
                i, acc_spmv.gpu_ms, acc_dot.gpu_ms,
                acc_axpy.gpu_ms, acc_scale.gpu_ms);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d solves\n",
            total_gpu_ms, num_iters);

    // Verification (run_cg reuses the timer, so destroy it afterwards)
    {
        run_cg();
        HIP_CHECK(hipDeviceSynchronize());

        spmv_kernel<<<gridSize, block_size>>>(d_row_ptr, d_col_idx, d_values, d_x, d_Ap, N);
        HIP_CHECK(hipMemcpy(d_r, d_b, vec_bytes, hipMemcpyDeviceToDevice));
        axpy_kernel<<<gridSize, block_size>>>(-1.0f, d_Ap, d_r, N);

        dot_product_kernel<<<numBlocks, block_size, smem>>>(d_r, d_r, d_partial, N);
        float res_norm = sqrtf(finish_dot(d_partial, numBlocks));

        dot_product_kernel<<<numBlocks, block_size, smem>>>(d_b, d_b, d_partial, N);
        float b_norm = sqrtf(finish_dot(d_partial, numBlocks));

        float rel_res = res_norm / fmaxf(b_norm, 1e-20f);
        fprintf(stderr, "Residual ||b-Ax||/||b|| = %.6e\n", rel_res);

        if (rel_res < 1.0f)
            fprintf(stderr, "PASS\n");
        else
            fprintf(stderr, "FAIL: relative residual too large\n");
    }

    bench_timer_destroy(&timer);

    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_p));
    HIP_CHECK(hipFree(d_Ap));
    HIP_CHECK(hipFree(d_partial));

    return 0;
}
