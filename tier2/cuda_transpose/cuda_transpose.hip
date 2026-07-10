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
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_tile_dim
//   BENCH_PARAM_warmup   (warmup launches per variant, default 3)
//   BENCH_PARAM_iteration    (timed launches per variant, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// of each variant (transpose_naive, transpose_optimized) carrying the GPU
// event time and host wall-clock time, then a summary whose total_time_ms is
// the SUM of all iterations' GPU event times. Effective bandwidth
// (2*N*N*sizeof(float) / time) is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
    int N          = 4096;
    int block_size = 256;
    int tile_dim   = 32;

    // Command-line args as fallback
    N          = parseIntParam(argc, argv, "--size", N);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);
    tile_dim   = parseIntParam(argc, argv, "--tile_dim", tile_dim);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_tile_dim");
    if (env_val) tile_dim = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    size_t num_elems = (size_t)N * N;
    size_t bytes     = num_elems * sizeof(float);

    // Print device info to stderr
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Matrix size: %d×%d  |  Warmup: %d  |  Timed: %d\n\n",
            N, N, num_warmup, num_iters);

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
             "\"size\":%d,\"block_size\":%d,\"tile_dim\":%d",
             N, block_size, tile_dim);

    // Both variants read d_idata (never modified) and overwrite d_odata in
    // full; repeating a launch performs identical work regardless of the
    // previous output values, so no per-iteration reset is needed.

    // -----------------------------------------------------------------------
    // Variant 1: Naive transpose
    // -----------------------------------------------------------------------
    HIP_CHECK(hipMemset(d_odata, 0, bytes));

    double total_gpu_ms =
        bench_run_loop("transpose_naive", num_warmup, num_iters, params, [&]() {
            transpose_naive<<<grid, block>>>(d_idata, d_odata, N, N);
        });

    // Verify naive transpose
    HIP_CHECK(hipMemcpy(h_odata, d_odata, bytes, hipMemcpyDeviceToHost));
    transpose_cpu(h_idata, h_ref, N, N);

    int naive_errors = 0;
    for (int i = 0; i < 1000 && i < (int)num_elems; ++i) {
        if (fabsf(h_odata[i] - h_ref[i]) > 1e-5f) {
            if (naive_errors < 5) {
                fprintf(stderr, "  Naive mismatch at %d: got %f, expected %f\n",
                        i, h_odata[i], h_ref[i]);
            }
            naive_errors++;
        }
    }
    fprintf(stderr, "  Verification: %s\n",
            (naive_errors == 0) ? "PASS" : "FAIL");

    // -----------------------------------------------------------------------
    // Variant 2: Optimized transpose (shared memory + bank-conflict-free)
    // -----------------------------------------------------------------------
    HIP_CHECK(hipMemset(d_odata, 0, bytes));

    total_gpu_ms +=
        bench_run_loop("transpose_optimized", num_warmup, num_iters, params, [&]() {
            transpose_optimized<<<grid, block>>>(d_idata, d_odata, N, N);
        });

    // Verify optimized transpose
    HIP_CHECK(hipMemcpy(h_odata, d_odata, bytes, hipMemcpyDeviceToHost));

    int opt_errors = 0;
    for (int i = 0; i < 1000 && i < (int)num_elems; ++i) {
        if (fabsf(h_odata[i] - h_ref[i]) > 1e-5f) {
            if (opt_errors < 5) {
                fprintf(stderr, "  Optimized mismatch at %d: got %f, expected %f\n",
                        i, h_odata[i], h_ref[i]);
            }
            opt_errors++;
        }
    }
    fprintf(stderr, "  Verification: %s\n",
            (opt_errors == 0) ? "PASS" : "FAIL");

    // -----------------------------------------------------------------------
    // Summary: sum of every recorded iteration's GPU event time
    // -----------------------------------------------------------------------
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations per variant\n",
            total_gpu_ms, num_iters);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    HIP_CHECK(hipFree(d_idata));
    HIP_CHECK(hipFree(d_odata));
    free(h_idata);
    free(h_odata);
    free(h_ref);

    return 0;
}
