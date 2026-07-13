// shoc_stencil2d.hip — SHOC 9-point 2D stencil benchmark (HIP, self-contained)
//
// Applies a 9-point star stencil iteratively on an N×N grid:
//
//   out[i][j] = 0.5  * in[i][j]                                   (center)
//             + 0.1  * (in[i-1][j] + in[i+1][j] +                (N/S/E/W)
//                       in[i][j-1] + in[i][j+1])
//             + 0.025 * (in[i-1][j-1] + in[i-1][j+1] +           (diagonals)
//                        in[i+1][j-1] + in[i+1][j+1])
//
// Interior cells only: rows/columns 1 .. N-2.
// Boundaries (row 0, row N-1, col 0, col N-1) remain fixed at 0.
// Uses ping-pong buffers: in_buf and out_buf are swapped each step.
//
// Default: N=2048, ITERATIONS=5
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./shoc_stencil2d [--n N] [--iterations I]
//
//   --n N            Grid dimension (N×N)                     (default: 2048)
//   --iterations I   Stencil steps in the timed run           (default: 5)
//
// Environment variables (override argv):
//   BENCH_PARAM_n, BENCH_PARAM_iterations, BENCH_PARAM_block_size
//
// Output (stdout): for the single timed run, one {"type":"kernel",...} JSON
// line for stencil2d_kernel carrying the accumulated GPU event time (time_ms)
// and host time (host_time_ms) of all stencil steps within the run, then a
// summary whose total_time_ms equals that record's GPU event time.
// Bandwidth (GB/s) is derived downstream from time_ms and params.
// Output (stderr): device info, timing details, correctness check

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static int parseIntParam(int argc, char** argv, const char* name, int defVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defVal;
}

// ---------------------------------------------------------------------------
// Stencil kernel — 9-point star stencil, 16×16 thread blocks
// ---------------------------------------------------------------------------

#define BLOCK_DIM 16

__global__ void stencil2d_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int N)
{
    // Map thread to interior cell (1-indexed)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  // col
    if (i >= N - 1 || j >= N - 1) return;

    float center = in[i * N + j];
    float north  = in[(i - 1) * N + j];
    float south  = in[(i + 1) * N + j];
    float west   = in[i * N + (j - 1)];
    float east   = in[i * N + (j + 1)];
    float nw     = in[(i - 1) * N + (j - 1)];
    float ne     = in[(i - 1) * N + (j + 1)];
    float sw     = in[(i + 1) * N + (j - 1)];
    float se     = in[(i + 1) * N + (j + 1)];

    out[i * N + j] = 0.5f   * center
                   + 0.1f   * (north + south + west + east)
                   + 0.025f * (nw + ne + sw + se);
}

// ---------------------------------------------------------------------------
// CPU reference (single iteration, for correctness check)
// ---------------------------------------------------------------------------

static void cpu_stencil2d(const float* in, float* out, int N)
{
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            float center = in[i * N + j];
            float north  = in[(i - 1) * N + j];
            float south  = in[(i + 1) * N + j];
            float west   = in[i * N + (j - 1)];
            float east   = in[i * N + (j + 1)];
            float nw     = in[(i - 1) * N + (j - 1)];
            float ne     = in[(i - 1) * N + (j + 1)];
            float sw     = in[(i + 1) * N + (j - 1)];
            float se     = in[(i + 1) * N + (j + 1)];

            out[i * N + j] = 0.5f   * center
                           + 0.1f   * (north + south + west + east)
                           + 0.025f * (nw + ne + sw + se);
        }
    }
}

// ---------------------------------------------------------------------------
// Correctness check on a small grid
// ---------------------------------------------------------------------------

static bool correctness_check()
{
    const int NC = 64;
    size_t bytes = (size_t)NC * NC * sizeof(float);

    std::vector<float> h_in(NC * NC, 0.0f);
    std::vector<float> h_out_cpu(NC * NC, 0.0f);
    std::vector<float> h_out_gpu(NC * NC, 0.0f);

    srand(42);
    for (int i = 0; i < NC * NC; ++i)
        h_in[i] = (float)(rand() % 100) / 10.0f;
    // Keep boundaries at 0
    for (int j = 0; j < NC; ++j) h_in[0 * NC + j] = 0.0f;
    for (int j = 0; j < NC; ++j) h_in[(NC - 1) * NC + j] = 0.0f;
    for (int i = 0; i < NC; ++i) h_in[i * NC + 0] = 0.0f;
    for (int i = 0; i < NC; ++i) h_in[i * NC + (NC - 1)] = 0.0f;

    // CPU reference
    cpu_stencil2d(h_in.data(), h_out_cpu.data(), NC);

    // GPU
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in,  h_in.data(),      bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((NC - 2 + BLOCK_DIM - 1) / BLOCK_DIM,
              (NC - 2 + BLOCK_DIM - 1) / BLOCK_DIM);
    stencil2d_kernel<<<grid, block>>>(d_in, d_out, NC);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out_gpu.data(), d_out, bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    bool pass = true;
    for (int i = 1; i < NC - 1; ++i) {
        for (int j = 1; j < NC - 1; ++j) {
            float ref = h_out_cpu[i * NC + j];
            float gpu = h_out_gpu[i * NC + j];
            float rel = fabsf(gpu - ref) / (fabsf(ref) + 1e-6f);
            if (rel > 1e-4f) {
                fprintf(stderr, "MISMATCH at [%d][%d]: gpu=%.6f cpu=%.6f rel=%.4e\n",
                        i, j, gpu, ref, rel);
                pass = false;
                if (!pass) goto done; // report first mismatch only
            }
        }
    }
done:
    return pass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    int N      = parseIntParam(argc, argv, "--n",          2048);
    int iters  = parseIntParam(argc, argv, "--iterations", 5);
    int block_size = 16;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iterations");
    if (env_val) iters = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d  |  Stencil steps/pass: %d\n\n",
            N, N, iters);

    // Correctness check runs AFTER the timed pass (below) — launching the
    // kernel-under-test on a 64×64 grid here would pre-warm the I-cache/module
    // state the single cold launch is meant to pay.

    // Allocate host and device buffers for full run
    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h_data(N * N, 0.0f);

    // Initialize interior with random data
    srand(123);
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j)
            h_data[i * N + j] = (float)(rand() % 100) / 10.0f;

    float *d_buf0, *d_buf1;
    HIP_CHECK(hipMalloc(&d_buf0, bytes));
    HIP_CHECK(hipMalloc(&d_buf1, bytes));
    HIP_CHECK(hipMemcpy(d_buf0, h_data.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_buf1, 0, bytes));

    // Kernel launch config
    dim3 block(block_size, block_size);
    dim3 grid((N - 2 + block_size - 1) / block_size,
              (N - 2 + block_size - 1) / block_size);

    char params[128];
    snprintf(params, sizeof(params),
             "\"n\":%d,\"iterations\":%d,\"block_size\":%d",
             N, iters, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;

    // One full stencil pass: `iters` ping-pong steps, each launch bracketed.
    // The buffers are set to the pristine input at the start of the run
    // (untimed) so the stencil is applied to well-defined data.
    auto run_pass = [&]() {
        bench_lap_reset(&acc);

        HIP_CHECK(hipMemcpy(d_buf0, h_data.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_buf1, 0, bytes));

        float *in_ptr = d_buf0, *out_ptr = d_buf1;
        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int it = 0; it < iters; ++it) {
            bench_lap_add(&acc, bench_time_op(&timer, [&]() {
                stencil2d_kernel<<<grid, block>>>(in_ptr, out_ptr, N);
            }));
            float* tmp = in_ptr; in_ptr = out_ptr; out_ptr = tmp;
        }
    };

    // Timed pass — one record with the accumulated kernel time
    double total_gpu_ms = 0.0;
    run_pass();
    bench_emit_iter("stencil2d_kernel", 0, &acc, params);
    total_gpu_ms += acc.gpu_ms;
    fprintf(stderr, "gpu = %.3f ms, host = %.3f ms (%d stencil steps)\n",
            acc.gpu_ms, acc.host_ms, iters);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Cleanup
    HIP_CHECK(hipFree(d_buf0));
    HIP_CHECK(hipFree(d_buf1));

    // Correctness check on a separate 64×64 grid, AFTER the timed pass so its
    // kernel launch cannot pre-warm the timed one. All JSON records and the
    // summary line have been emitted by now, so failing here only marks the run
    // as failed without losing the measurements.
    bool ok = correctness_check();
    fprintf(stderr, "Correctness check (64×64, 1 iteration): %s\n",
            ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
