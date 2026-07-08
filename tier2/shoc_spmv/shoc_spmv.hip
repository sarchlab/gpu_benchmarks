// shoc_spmv.hip — SHOC CSR Sparse Matrix-Vector Multiply benchmark (HIP, self-contained)
//
// Computes: y = A * x
//   where A is a synthetic CSR sparse matrix with exactly nnz_per_row non-zeros per row.
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback.
//
// Environment variables:
//   BENCH_PARAM_num_rows, BENCH_PARAM_nnz_per_row, BENCH_PARAM_precision
//   BENCH_PARAM_num_warmup   (warmup launches, default 3)
//   BENCH_PARAM_num_iters    (timed launches, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// carrying the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the SUM of all iterations' GPU event times. Bandwidth and
// GFLOPS are derived downstream from time_ms and params.
// Output (stderr): Human-readable diagnostics

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

static const char* parseStrParam(int argc, char** argv, const char* name,
                                 const char* defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// CSR SpMV kernel: each thread handles one row
// ---------------------------------------------------------------------------

__global__ void spmv_csr_kernel(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    float sum = 0.0f;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    for (int k = start; k < end; ++k)
        sum += values[k] * x[col_idx[k]];
    y[row] = sum;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static void cpu_spmv(const int* row_ptr, const int* col_idx,
                     const float* values, const float* x,
                     float* y, int num_rows)
{
    for (int row = 0; row < num_rows; ++row) {
        float sum = 0.0f;
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
            sum += values[k] * x[col_idx[k]];
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int num_rows    = 16384;
    int nnz_per_row = 32;
    const char* precision = "float";

    // Command-line fallback
    num_rows    = parseIntParam(argc, argv, "--num_rows", num_rows);
    nnz_per_row = parseIntParam(argc, argv, "--nnz_per_row", nnz_per_row);
    precision   = parseStrParam(argc, argv, "--precision", precision);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_num_rows");
    if (env_val) num_rows = atoi(env_val);
    env_val = getenv("BENCH_PARAM_nnz_per_row");
    if (env_val) nnz_per_row = atoi(env_val);
    /* Validate string params against their known sets, so the values echoed
     * into the JSON records are always canonical literals. */
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) precision = env_val;
    if (strcmp(precision, "float") != 0 && strcmp(precision, "double") != 0 &&
        strcmp(precision, "half") != 0) {
        fprintf(stderr, "Invalid precision '%s' (use float, double, half)\n",
                precision);
        return EXIT_FAILURE;
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int num_cols   = num_rows;
    int nnz        = num_rows * nnz_per_row;
    int BLOCK_SIZE = 256;

    // Allocate host arrays
    int*   h_row_ptr = (int*)  malloc((size_t)(num_rows + 1) * sizeof(int));
    int*   h_col_idx = (int*)  malloc((size_t)nnz            * sizeof(int));
    float* h_values  = (float*)malloc((size_t)nnz            * sizeof(float));
    float* h_x       = (float*)malloc((size_t)num_rows       * sizeof(float));
    float* h_y       = (float*)malloc((size_t)num_rows       * sizeof(float));
    float* h_y_ref   = (float*)malloc((size_t)num_rows       * sizeof(float));

    // Build CSR matrix
    srand(42);
    for (int i = 0; i <= num_rows; ++i)
        h_row_ptr[i] = i * nnz_per_row;

    for (int k = 0; k < nnz; ++k) {
        h_col_idx[k] = rand() % num_cols;
        h_values[k]  = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < num_rows; ++i)
        h_x[i] = (float)rand() / (float)RAND_MAX;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Rows: %d  |  NNZ_per_row: %d  |  Total NNZ: %d\n\n",
            num_rows, nnz_per_row, nnz);

    // Device allocations
    int*   d_row_ptr;
    int*   d_col_idx;
    float* d_values;
    float* d_x;
    float* d_y;
    HIP_CHECK(hipMalloc(&d_row_ptr, (size_t)(num_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx, (size_t)nnz            * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_values,  (size_t)nnz            * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x,       (size_t)num_rows       * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y,       (size_t)num_rows       * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr,
                        (size_t)(num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, h_col_idx,
                        (size_t)nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values, h_values,
                        (size_t)nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x,
                        (size_t)num_rows * sizeof(float), hipMemcpyHostToDevice));

    int grid_size = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    char params[160];
    snprintf(params, sizeof(params),
             "\"num_rows\":%d,\"nnz_per_row\":%d,\"precision\":\"%s\"",
             num_rows, nnz_per_row, precision);

    // Warmup + timed iterations, each recorded individually. The kernel only
    // reads the matrix and x (never modified) and fully overwrites y, so
    // repeated launches perform identical work — no per-iteration reset.
    double total_gpu_ms =
        bench_run_loop("spmv_csr_kernel", num_warmup, num_iters, params, [&]() {
            spmv_csr_kernel<<<grid_size, BLOCK_SIZE>>>(
                d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
        });

    // Copy result back
    HIP_CHECK(hipMemcpy(h_y, d_y, (size_t)num_rows * sizeof(float), hipMemcpyDeviceToHost));

    // CPU reference
    cpu_spmv(h_row_ptr, h_col_idx, h_values, h_x, h_y_ref, num_rows);

    bool pass = true;
    int check_n = 32;
    for (int i = 0; i < check_n && i < num_rows; ++i) {
        float ref = h_y_ref[i];
        float rel = fabsf(h_y[i] - ref) / (fabsf(ref) + 1e-6f);
        if (rel > 1e-4f) {
            fprintf(stderr, "MISMATCH y[%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                    i, h_y[i], ref, rel);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (first %d elements): %s\n\n",
            check_n, pass ? "PASS" : "FAIL");

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    free(h_row_ptr);
    free(h_col_idx);
    free(h_values);
    free(h_x);
    free(h_y);
    free(h_y_ref);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
