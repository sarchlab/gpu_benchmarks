// shoc_spmv.hip — SHOC CSR Sparse Matrix-Vector Multiply benchmark (HIP, self-contained)
//
// Computes: y = A * x
//   where A is a synthetic CSR sparse matrix with exactly NNZ_PER_ROW non-zeros per row.
//
// Parameters (compile-time constants):
//   NUM_ROWS    = 16384
//   NNZ_PER_ROW = 32
//   NNZ         = NUM_ROWS * NNZ_PER_ROW = 524288
//
// GPU kernel: each thread handles one row (row-per-thread CSR SpMV).
//
// Metrics:
//   GB/s   = (NNZ * (sizeof(float) + sizeof(int)) + (NUM_ROWS+1)*sizeof(int)
//             + NUM_ROWS*sizeof(float) + NNZ*sizeof(float)) / time_s / 1e9
//   GFLOPS = 2 * NNZ / time_s / 1e9
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Output (stdout): CSV — spmv,<NUM_ROWS>,<NNZ_PER_ROW>,<time_ms>,<GBs>,<GFLOPS>
// Output (stderr): device info, per-iteration times, verification result

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Benchmark constants
// ---------------------------------------------------------------------------

#define NUM_ROWS    16384
#define NNZ_PER_ROW 32
#define NUM_COLS    NUM_ROWS
#define NNZ         (NUM_ROWS * NNZ_PER_ROW)
#define BLOCK_SIZE  256
#define ITERATIONS  5

// ---------------------------------------------------------------------------
// Inlined utilities
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd)                                                          \
    do {                                                                        \
        hipError_t _e = (cmd);                                                  \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP error %s at %s:%d\n",                         \
                    hipGetErrorString(_e), __FILE__, __LINE__);                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct BenchmarkTimer {
    hipEvent_t start, stop;
    BenchmarkTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~BenchmarkTimer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    void record_start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start, stream));
    }
    void record_stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

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
// CPU reference for correctness check
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
    const int num_rows    = NUM_ROWS;
    const int nnz_per_row = NNZ_PER_ROW;
    const int nnz         = NNZ;
    const int iters       = ITERATIONS;

    // Allocate host arrays
    int*   h_row_ptr = (int*)  malloc((size_t)(num_rows + 1) * sizeof(int));
    int*   h_col_idx = (int*)  malloc((size_t)nnz            * sizeof(int));
    float* h_values  = (float*)malloc((size_t)nnz            * sizeof(float));
    float* h_x       = (float*)malloc((size_t)num_rows       * sizeof(float));
    float* h_y       = (float*)malloc((size_t)num_rows       * sizeof(float));
    float* h_y_ref   = (float*)malloc((size_t)num_rows       * sizeof(float));

    // Build CSR matrix: exactly nnz_per_row non-zeros per row, random column indices
    srand(42);
    for (int i = 0; i <= num_rows; ++i)
        h_row_ptr[i] = i * nnz_per_row;

    for (int k = 0; k < nnz; ++k) {
        h_col_idx[k] = rand() % NUM_COLS;
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
    fprintf(stderr, "Rows: %d  |  NNZ_per_row: %d  |  Total NNZ: %d  |  Iterations: %d\n\n",
            num_rows, nnz_per_row, nnz, iters);

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

    BenchmarkTimer timer;

    // Warmup
    hipLaunchKernelGGL(spmv_csr_kernel, grid_size, BLOCK_SIZE, 0, 0,
                       d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations
    std::vector<double> times(iters);
    for (int iter = 0; iter < iters; ++iter) {
        timer.record_start();
        hipLaunchKernelGGL(spmv_csr_kernel, grid_size, BLOCK_SIZE, 0, 0,
                           d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
        timer.record_stop();
        times[iter] = static_cast<double>(timer.elapsed_ms());
        fprintf(stderr, "  iter %d: %.4f ms\n", iter, times[iter]);
    }

    // Copy result back
    HIP_CHECK(hipMemcpy(h_y, d_y, (size_t)num_rows * sizeof(float), hipMemcpyDeviceToHost));

    // CPU reference (verify first 32 elements)
    cpu_spmv(h_row_ptr, h_col_idx, h_values, h_x, h_y_ref, num_rows);

    bool pass = true;
    int check_n = 32;
    for (int i = 0; i < check_n; ++i) {
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

    // Compute statistics
    double sum_ms = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iters; ++i) {
        sum_ms += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg_ms = sum_ms / iters;

    double variance = 0.0;
    for (int i = 0; i < iters; ++i) {
        double d = times[i] - avg_ms;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;

    // GB/s: values(float) + col_idx(int) + row_ptr(int) + x(float) + y(float)
    // = nnz*8 + (num_rows+1)*4 + num_rows*4 + num_rows*4
    double bytes = (double)nnz * (sizeof(float) + sizeof(int))
                 + (double)(num_rows + 1) * sizeof(int)
                 + (double)num_rows * sizeof(float)
                 + (double)num_rows * sizeof(float);
    double gbps   = bytes / (avg_ms * 1e-3) / 1e9;
    double gflops = 2.0 * (double)nnz / (avg_ms * 1e-3) / 1e9;

    fprintf(stderr, "GB/s: %.2f  GFLOPS: %.2f  (avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gbps, gflops, avg_ms, mn, mx, stddev);

    // CSV output: spmv,<NUM_ROWS>,<NNZ_PER_ROW>,<time_ms>,<GBs>,<GFLOPS>
    printf("spmv,%d,%d,%.4f,%.2f,%.2f\n",
           num_rows, nnz_per_row, avg_ms, gbps, gflops);

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
