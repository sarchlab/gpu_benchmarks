// cuda_transpose.hip — Shared-memory tiled matrix transpose benchmark (HIP)
//
// Two variants on a 4096×4096 single-precision matrix:
//   1. Naive transpose  — coalesced reads, strided writes
//   2. Optimized transpose — shared memory tile with +1 padding (bank-conflict-free)
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./cuda_transpose [--size N]
//
//   --size N  Matrix dimension N (N×N)  (default: 4096)
//
// Environment variables (override argv):
//   BENCH_PARAM_size
//
// Output (stdout): one {"type":"kernel",...} JSON line per variant
// (transpose_naive, transpose_optimized), each timed exactly once, carrying
// the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the sum of all recorded GPU event times. Effective bandwidth
// (2*N*N*sizeof(float) / time) is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static int parseIntParam(int argc, char** argv, const char* name,
                         int defaultVal) {
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
// Constants — classic CUDA transpose pattern
// ---------------------------------------------------------------------------

#define TILE_DIM   32
#define BLOCK_ROWS  8

// ---------------------------------------------------------------------------
// Kernel 1: Naive transpose
// Each thread reads one element (coalesced) and writes one element (strided).
// ---------------------------------------------------------------------------

__global__ void transpose_naive(const float* __restrict__ idata,
                                float* __restrict__ odata,
                                int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + j) < height) {
            odata[xIndex * height + (yIndex + j)] =
                idata[(yIndex + j) * width + xIndex];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Optimized transpose with shared memory (+1 padding)
// Coalesced read into shared memory, __syncthreads, coalesced write out.
// The +1 padding on the shared memory tile avoids bank conflicts.
// ---------------------------------------------------------------------------

__global__ void transpose_optimized(const float* __restrict__ idata,
                                    float* __restrict__ odata,
                                    int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from global memory (coalesced reads)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] =
                idata[(yIndex + j) * width + xIndex];
        }
    }

    __syncthreads();

    // Write transposed tile to global memory (coalesced writes)
    // Swap block indices for transposed output position
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < height && (yIndex + j) < width) {
            odata[(yIndex + j) * height + xIndex] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference transpose (for verification)
// ---------------------------------------------------------------------------

static void transpose_cpu(const float* in, float* out, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out[x * height + y] = in[y * width + x];
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Default values from params.json
    int N = 4096;

    // Command-line args as fallback
    N = parseIntParam(argc, argv, "--size", N);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);

    if (N <= 0) {
        fprintf(stderr, "Transpose size must be positive\n");
        return EXIT_FAILURE;
    }
    // Both GPU variants and the CPU reference use signed int flat indices.
    if ((unsigned long long)N * N > INT_MAX) {
        fprintf(stderr, "Transpose size exceeds the signed index range\n");
        return EXIT_FAILURE;
    }

    size_t num_elems = (size_t)N * N;
    size_t bytes     = num_elems * sizeof(float);

    // Print device info to stderr
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d\n\n", N, N);

    // Host allocations
    float* h_idata = (float*)malloc(bytes);
    float* h_odata = (float*)malloc(bytes);
    float* h_ref   = (float*)malloc(bytes);

    // Initialize input matrix
    for (size_t i = 0; i < num_elems; ++i) {
        h_idata[i] = (float)(i % 1000) * 0.001f;
    }

    // Device allocations
    float *d_idata, *d_odata;
    HIP_CHECK(hipMalloc(&d_idata, bytes));
    HIP_CHECK(hipMalloc(&d_odata, bytes));
    HIP_CHECK(hipMemcpy(d_idata, h_idata, bytes, hipMemcpyHostToDevice));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"verification\":\"full_cpu_v1\"", N);

    // Both variants read d_idata (never modified) and overwrite d_odata in
    // full; repeating a launch performs identical work regardless of the
    // previous output values, so no per-iteration reset is needed.

    // -----------------------------------------------------------------------
    // Variant 1: Naive transpose
    // -----------------------------------------------------------------------
    HIP_CHECK(hipMemset(d_odata, 0, bytes));

    double total_gpu_ms =
        bench_run_once("transpose_naive", params, [&]() {
            transpose_naive<<<grid, block>>>(d_idata, d_odata, N, N);
        });

    // Verify naive transpose
    HIP_CHECK(hipMemcpy(h_odata, d_odata, bytes, hipMemcpyDeviceToHost));
    transpose_cpu(h_idata, h_ref, N, N);

    size_t naive_errors = 0;
    for (size_t i = 0; i < num_elems; ++i) {
        if (!std::isfinite(h_odata[i]) || !std::isfinite(h_ref[i]) ||
            fabsf(h_odata[i] - h_ref[i]) > 1e-5f) {
            if (naive_errors < 5) {
                fprintf(stderr, "  Naive mismatch at %zu: got %f, expected %f\n",
                        i, h_odata[i], h_ref[i]);
            }
            naive_errors++;
        }
    }
    fprintf(stderr, "  Naive verification: %s (%zu mismatches in %zu elements)\n",
            (naive_errors == 0) ? "PASS" : "FAIL", naive_errors, num_elems);
    bool verify_ok = (naive_errors == 0);

    // -----------------------------------------------------------------------
    // Variant 2: Optimized transpose (shared memory + bank-conflict-free)
    // -----------------------------------------------------------------------
    HIP_CHECK(hipMemset(d_odata, 0, bytes));

    total_gpu_ms +=
        bench_run_once("transpose_optimized", params, [&]() {
            transpose_optimized<<<grid, block>>>(d_idata, d_odata, N, N);
        });

    // Verify optimized transpose
    HIP_CHECK(hipMemcpy(h_odata, d_odata, bytes, hipMemcpyDeviceToHost));

    size_t opt_errors = 0;
    for (size_t i = 0; i < num_elems; ++i) {
        if (!std::isfinite(h_odata[i]) || !std::isfinite(h_ref[i]) ||
            fabsf(h_odata[i] - h_ref[i]) > 1e-5f) {
            if (opt_errors < 5) {
                fprintf(stderr, "  Optimized mismatch at %zu: got %f, expected %f\n",
                        i, h_odata[i], h_ref[i]);
            }
            opt_errors++;
        }
    }
    fprintf(stderr, "  Optimized verification: %s (%zu mismatches in %zu elements)\n",
            (opt_errors == 0) ? "PASS" : "FAIL", opt_errors, num_elems);
    if (opt_errors != 0) verify_ok = false;

    // -----------------------------------------------------------------------
    // Summary: sum of every recorded iteration's GPU event time
    // -----------------------------------------------------------------------
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    HIP_CHECK(hipFree(d_idata));
    HIP_CHECK(hipFree(d_odata));
    free(h_idata);
    free(h_odata);
    free(h_ref);

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
