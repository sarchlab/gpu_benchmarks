// cusparse_spmv.hip — CSR Sparse Matrix-Vector Multiply benchmark
//   using cuSPARSE (NVIDIA) or rocSPARSE (AMD) vendor library APIs.
//
// Computes: y = A * x
//   where A is a synthetic CSR sparse matrix with exactly NNZ_PER_ROW non-zeros per row.
//
// Parameters (from env vars BENCH_PARAM_*):
//   num_rows    — number of rows (and columns) in the square sparse matrix
//   nnz_per_row — non-zeros per row
//   warmup      — warmup iterations
//   iters       — timed iterations
//
// Metrics:
//   GB/s   = NNZ * 12.0 / time_s / 1e9
//   GFLOPS = 2.0 * NNZ  / time_s / 1e9
//
// Output (stdout): JSON-lines protocol
// Output (stderr): device info, per-iteration times, verification result
//
// Build:
//   CUDA:  nvcc -O3 -std=c++14 -x cu -DUSE_CUSPARSE cusparse_spmv.hip -lcusparse -o cusparse_spmv
//   ROCm:  hipcc -O3 -std=c++14 -DUSE_ROCSPARSE cusparse_spmv.hip -lrocsparse -o cusparse_spmv

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Platform headers and macros
// ---------------------------------------------------------------------------

#ifdef USE_CUSPARSE
#include <cusparse.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do {                                                 \
    cudaError_t _e = (x);                                                  \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                cudaGetErrorString(_e), __FILE__, __LINE__);               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

#define SP_CHECK(x) do {                                                   \
    cusparseStatus_t _e = (x);                                             \
    if (_e != CUSPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSPARSE error %d at %s:%d\n",                   \
                (int)_e, __FILE__, __LINE__);                              \
        exit(1);                                                           \
    }                                                                      \
} while (0)

#define KERNEL_NAME "cusparseSpMV"

#elif defined(USE_ROCSPARSE)
#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(x) do {                                                  \
    hipError_t _e = (x);                                                   \
    if (_e != hipSuccess) {                                                \
        fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                hipGetErrorString(_e), __FILE__, __LINE__);                \
        exit(1);                                                           \
    }                                                                      \
} while (0)

#define SP_CHECK(x) do {                                                   \
    rocsparse_status _e = (x);                                             \
    if (_e != rocsparse_status_success) {                                  \
        fprintf(stderr, "rocSPARSE error %d at %s:%d\n",                  \
                (int)_e, __FILE__, __LINE__);                              \
        exit(1);                                                           \
    }                                                                      \
} while (0)

#define KERNEL_NAME "rocsparse_scsrmv"

#else
#error "Define USE_CUSPARSE or USE_ROCSPARSE"
#endif

// ---------------------------------------------------------------------------
// CPU SpMV reference for correctness verification
// ---------------------------------------------------------------------------

static void cpu_spmv(int num_rows, const int *row_ptr, const int *col_idx,
                     const float *values, const float *x, float *y)
{
    for (int i = 0; i < num_rows; i++) {
        float s = 0.0f;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++)
            s += values[k] * x[col_idx[k]];
        y[i] = s;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // -----------------------------------------------------------------------
    // Default parameters (overridden by BENCH_PARAM_* env vars)
    // -----------------------------------------------------------------------
    int num_rows    = 16384;
    int nnz_per_row = 32;
    int warmup      = 5;
    int iters       = 5;

    const char *env_val;
    env_val = getenv("BENCH_PARAM_num_rows");
    if (env_val) num_rows = atoi(env_val);
    env_val = getenv("BENCH_PARAM_nnz_per_row");
    if (env_val) nnz_per_row = atoi(env_val);
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) warmup = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) iters = atoi(env_val);

    const int num_cols = num_rows;
    const int nnz      = num_rows * nnz_per_row;

    // -----------------------------------------------------------------------
    // Allocate host arrays
    // -----------------------------------------------------------------------
    int   *h_row_ptr = (int *)  malloc((size_t)(num_rows + 1) * sizeof(int));
    int   *h_col_idx = (int *)  malloc((size_t)nnz            * sizeof(int));
    float *h_values  = (float *)malloc((size_t)nnz            * sizeof(float));
    float *h_x       = (float *)malloc((size_t)num_cols       * sizeof(float));
    float *h_y       = (float *)malloc((size_t)num_rows       * sizeof(float));
    float *h_y_ref   = (float *)malloc((size_t)num_rows       * sizeof(float));

    if (!h_row_ptr || !h_col_idx || !h_values || !h_x || !h_y || !h_y_ref) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // -----------------------------------------------------------------------
    // Build CSR matrix: exactly nnz_per_row non-zeros per row
    // -----------------------------------------------------------------------
    srand(42);
    for (int i = 0; i <= num_rows; i++)
        h_row_ptr[i] = i * nnz_per_row;

    for (int k = 0; k < nnz; k++) {
        h_col_idx[k] = rand() % num_cols;
        h_values[k]  = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < num_cols; i++)
        h_x[i] = (float)rand() / (float)RAND_MAX;

    // -----------------------------------------------------------------------
    // Platform-specific device setup, memory, SpMV execution
    // -----------------------------------------------------------------------

    float alpha = 1.0f;
    float beta  = 0.0f;

#ifdef USE_CUSPARSE
    // -------------------------------------------------------------------
    // CUDA / cuSPARSE path
    // -------------------------------------------------------------------
    {
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        fprintf(stderr, "Platform: cuSPARSE\n");
        fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    }
    fprintf(stderr, "Rows: %d  |  NNZ_per_row: %d  |  Total NNZ: %d\n"
                    "Warmup: %d  |  Iterations: %d\n\n",
            num_rows, nnz_per_row, nnz, warmup, iters);

    // Device allocations
    int   *d_row_ptr;
    int   *d_col_idx;
    float *d_values;
    float *d_x;
    float *d_y;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz            * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,  (size_t)nnz            * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,       (size_t)num_cols       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,       (size_t)num_rows       * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr,
                          (size_t)(num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx,
                          (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,  h_values,
                          (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x,
                          (size_t)num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)num_rows * sizeof(float)));

    // Create cuSPARSE handle
    cusparseHandle_t sp_handle;
    SP_CHECK(cusparseCreate(&sp_handle));

    // Create CSR matrix descriptor (modern API, CUDA 11+)
    cusparseSpMatDescr_t mat_A;
    SP_CHECK(cusparseCreateCsr(
        &mat_A,
        (int64_t)num_rows, (int64_t)num_cols, (int64_t)nnz,
        d_row_ptr, d_col_idx, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    cusparseDnVecDescr_t vec_x, vec_y;
    SP_CHECK(cusparseCreateDnVec(&vec_x, (int64_t)num_cols, d_x, CUDA_R_32F));
    SP_CHECK(cusparseCreateDnVec(&vec_y, (int64_t)num_rows, d_y, CUDA_R_32F));

    // Query workspace size
    size_t buffer_sz = 0;
    SP_CHECK(cusparseSpMV_bufferSize(
        sp_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_A, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_sz));

    void *d_buffer = NULL;
    if (buffer_sz > 0)
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_sz));

    // Timing events
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warmup
    for (int w = 0; w < warmup; w++) {
        SP_CHECK(cusparseSpMV(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_A, vec_x, &beta, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations
    double times[64];
    double sum_ms = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int iter = 0; iter < iters; iter++) {
        float ms_f = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev_start, 0));
        SP_CHECK(cusparseSpMV(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_A, vec_x, &beta, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer));
        CUDA_CHECK(cudaEventRecord(ev_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_f, ev_start, ev_stop));
        times[iter] = (double)ms_f;
        sum_ms += times[iter];
        if (times[iter] < mn) mn = times[iter];
        if (times[iter] > mx) mx = times[iter];
        fprintf(stderr, "  iter %d: %.4f ms\n", iter, times[iter]);
    }

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y, d_y, (size_t)num_rows * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify
    cpu_spmv(num_rows, h_row_ptr, h_col_idx, h_values, h_x, h_y_ref);
    bool pass = true;
    int check_n = 32;
    for (int i = 0; i < check_n; i++) {
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

    // Statistics
    double avg_ms = sum_ms / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; i++) {
        double d = times[i] - avg_ms;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;
    double time_s = avg_ms * 1e-3;
    double gbps   = (double)nnz * 12.0 / time_s / 1e9;
    double gflops = 2.0 * (double)nnz / time_s / 1e9;

    fprintf(stderr,
            "GB/s: %.2f  GFLOPS: %.2f  "
            "(avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gbps, gflops, avg_ms, mn, mx, stddev);

    // JSON-lines: kernel event
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"num_rows\":%d,\"nnz_per_row\":%d,\"warmup\":%d,\"iters\":%d}}\n",
           KERNEL_NAME, avg_ms, num_rows, nnz_per_row, warmup, iters);

    // JSON-lines: summary event
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f},{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           sum_ms, gbps, gflops);

    // Cleanup
    if (d_buffer) cudaFree(d_buffer);
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cusparseDestroy(sp_handle);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_row_ptr); free(h_col_idx); free(h_values);
    free(h_x); free(h_y); free(h_y_ref);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;

#else
    // -------------------------------------------------------------------
    // AMD ROCm / rocSPARSE path
    // -------------------------------------------------------------------
    {
        int device_id = 0;
        HIP_CHECK(hipGetDevice(&device_id));
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
        fprintf(stderr, "Platform: rocSPARSE\n");
        fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    }
    fprintf(stderr, "Rows: %d  |  NNZ_per_row: %d  |  Total NNZ: %d\n"
                    "Warmup: %d  |  Iterations: %d\n\n",
            num_rows, nnz_per_row, nnz, warmup, iters);

    // Device allocations
    int   *d_row_ptr;
    int   *d_col_idx;
    float *d_values;
    float *d_x;
    float *d_y;

    HIP_CHECK(hipMalloc(&d_row_ptr, (size_t)(num_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx, (size_t)nnz            * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_values,  (size_t)nnz            * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x,       (size_t)num_cols       * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y,       (size_t)num_rows       * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr,
                        (size_t)(num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, h_col_idx,
                        (size_t)nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values,  h_values,
                        (size_t)nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x,
                        (size_t)num_cols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_y, 0, (size_t)num_rows * sizeof(float)));

    // Create rocSPARSE handle and matrix descriptor
    rocsparse_handle    sp_handle;
    rocsparse_mat_descr mat_descr;

    SP_CHECK(rocsparse_create_handle(&sp_handle));
    SP_CHECK(rocsparse_create_mat_descr(&mat_descr));
    rocsparse_set_mat_index_base(mat_descr, rocsparse_index_base_zero);

    // Timing events
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    // Warmup
    for (int w = 0; w < warmup; w++) {
        SP_CHECK(rocsparse_scsrmv(
            sp_handle,
            rocsparse_operation_none,
            num_rows, num_cols, nnz,
            &alpha, mat_descr,
            d_values, d_row_ptr, d_col_idx,
            d_x, &beta, d_y));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations
    double times[64];
    double sum_ms = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int iter = 0; iter < iters; iter++) {
        float ms_f = 0.0f;
        HIP_CHECK(hipEventRecord(ev_start, 0));
        SP_CHECK(rocsparse_scsrmv(
            sp_handle,
            rocsparse_operation_none,
            num_rows, num_cols, nnz,
            &alpha, mat_descr,
            d_values, d_row_ptr, d_col_idx,
            d_x, &beta, d_y));
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&ms_f, ev_start, ev_stop));
        times[iter] = (double)ms_f;
        sum_ms += times[iter];
        if (times[iter] < mn) mn = times[iter];
        if (times[iter] > mx) mx = times[iter];
        fprintf(stderr, "  iter %d: %.4f ms\n", iter, times[iter]);
    }

    // Copy result back
    HIP_CHECK(hipMemcpy(h_y, d_y, (size_t)num_rows * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify
    cpu_spmv(num_rows, h_row_ptr, h_col_idx, h_values, h_x, h_y_ref);
    bool pass = true;
    int check_n = 32;
    for (int i = 0; i < check_n; i++) {
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

    // Statistics
    double avg_ms = sum_ms / iters;
    double variance = 0.0;
    for (int i = 0; i < iters; i++) {
        double d = times[i] - avg_ms;
        variance += d * d;
    }
    double stddev = (iters > 1) ? sqrt(variance / (iters - 1)) : 0.0;
    double time_s = avg_ms * 1e-3;
    double gbps   = (double)nnz * 12.0 / time_s / 1e9;
    double gflops = 2.0 * (double)nnz / time_s / 1e9;

    fprintf(stderr,
            "GB/s: %.2f  GFLOPS: %.2f  "
            "(avg %.4f ms, min %.4f ms, max %.4f ms, stddev %.4f ms)\n",
            gbps, gflops, avg_ms, mn, mx, stddev);

    // JSON-lines: kernel event
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"num_rows\":%d,\"nnz_per_row\":%d,\"warmup\":%d,\"iters\":%d}}\n",
           KERNEL_NAME, avg_ms, num_rows, nnz_per_row, warmup, iters);

    // JSON-lines: summary event
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f},{\"name\":\"GFLOPS\",\"value\":%.2f}]}\n",
           sum_ms, gbps, gflops);

    // Cleanup
    rocsparse_destroy_mat_descr(mat_descr);
    rocsparse_destroy_handle(sp_handle);
    hipEventDestroy(ev_start);
    hipEventDestroy(ev_stop);
    hipFree(d_row_ptr);
    hipFree(d_col_idx);
    hipFree(d_values);
    hipFree(d_x);
    hipFree(d_y);

    free(h_row_ptr); free(h_col_idx); free(h_values);
    free(h_x); free(h_y); free(h_y_ref);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
