// parboil_stencil.hip — Parboil 3D Jacobi Stencil benchmark (HIP, self-contained)
//
// Pinned Parboil 7-point operator using ping-pong buffers.
// One complete sequence of num_timesteps launches runs per process.
//
// Stencil: out = c1 * (z+1 + z-1 + y+1 + y-1 + x+1 + x-1 neighbors)
//                - c0 * center
// c0 = 1.0f/6.0f, c1 = 1.0f/6.0f/6.0f, as in Parboil CPU/cuda_base
// at abduld/Parboil commit a4c63da28f45699e9a274cbc49af3e067162aa2b.
//
// Boundary cells (1-cell border) remain fixed.
//
// Compiles with hipcc (AMD ROCm) or nvcc (CUDA).
//
// Usage:
//   ./parboil_stencil [--grid_dim N] [--num_timesteps T]
//
//   --grid_dim N        Grid dimension (N×N×N)           (default: 64)
//   --num_timesteps T   Stencil time-steps per pass      (default: 10)
//
// Environment variables (override argv):
//   BENCH_PARAM_grid_dim, BENCH_PARAM_num_timesteps
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line
// carrying the stencil kernel's accumulated GPU event time and host
// wall-clock time over the pass's num_timesteps launches, then a summary
// whose total_time_ms is the SUM of all recorded GPU event times.
// GFLOPS / GB/s are derived downstream from time_ms and params.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

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
// 7-point 3D stencil kernel
// ---------------------------------------------------------------------------

__global__ void stencil3d(const float* __restrict__ in,
                          float*       __restrict__ out,
                          int nx, int ny, int nz,
                          float c0, float c1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;

    // Only compute interior points (skip 1-cell boundary)
    if (ix >= 1 && ix < nx - 1 &&
        iy >= 1 && iy < ny - 1 &&
        iz >= 1 && iz < nz - 1) {
        int idx = iz * ny * nx + iy * nx + ix;
        // Preserve the pinned cuda_base expression and neighbor order.
        out[idx] = (in[idx + ny*nx] + in[idx - ny*nx]
                  + in[idx + nx] + in[idx - nx]
                  + in[idx + 1] + in[idx - 1]) * c1 - in[idx] * c0;
    }
}

// Both buffers must start with the input boundary. Return the latest state
// after exactly num_timesteps updates, including odd timestep counts.
static float* run_stencil_pass(float* src, float* dst, int nx, int ny, int nz,
                               int num_timesteps, bench_lap_t* elapsed) {
    const float c0 = 1.0f / 6.0f;
    const float c1 = 1.0f / 6.0f / 6.0f;
    const dim3 block(16, 16, 1);
    const dim3 grid((nx + 15) / 16, (ny + 15) / 16, nz);
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_reset(elapsed);
    bench_pass_warm();
    for (int t = 0; t < num_timesteps; ++t) {
        bench_lap_add(elapsed, bench_time_op(&timer, [&]() {
            stencil3d<<<grid, block>>>(src, dst, nx, ny, nz, c0, c1);
        }));
        float* tmp = src; src = dst; dst = tmp;
    }
    bench_timer_destroy(&timer);
    return src;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Default values from params.json
    int grid_dim      = 64;
    int num_timesteps = 10;

    // Command-line args as fallback
    grid_dim      = parseIntParam(argc, argv, "--grid_dim",      grid_dim);
    num_timesteps = parseIntParam(argc, argv, "--num_timesteps", num_timesteps);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_grid_dim");
    if (env_val) grid_dim = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_timesteps");
    if (env_val) num_timesteps = atoi(env_val);

    if (grid_dim < 3 || grid_dim > INT_MAX / grid_dim / grid_dim || num_timesteps < 1) {
        fprintf(stderr, "Require grid_dim>=3, grid_dim^3 within signed indexing, and num_timesteps>=1\n");
        return 1;
    }

    int nx = grid_dim;
    int ny = grid_dim;
    int nz = grid_dim;
    int total = nx * ny * nz;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d×%d  |  Timesteps/pass: %d\n\n",
            nx, ny, nz, num_timesteps);

    // Host allocation and initialization
    float* h_data = (float*)malloc(total * sizeof(float));
    if (!h_data) { fprintf(stderr, "malloc failed\n"); return 1; }
    srand(42);
    for (int i = 0; i < total; ++i) {
        h_data[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // Device allocations (ping-pong buffers)
    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, total * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_data, total * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_data, total * sizeof(float), hipMemcpyHostToDevice));

    char params[160];
    snprintf(params, sizeof(params),
             "\"grid_dim\":%d,\"num_timesteps\":%d",
             grid_dim, num_timesteps);

    // Initial copies precede one entry warmer. No resets or warmers occur
    // between the dependent timesteps; accumulate only their kernel times.
    bench_lap_t acc_stencil;
    run_stencil_pass(d_A, d_B, nx, ny, nz, num_timesteps, &acc_stencil);

    double total_gpu_ms = 0.0;
    bench_emit_iter("stencil3d", 0, &acc_stencil, params);
    total_gpu_ms += acc_stencil.gpu_ms;
    fprintf(stderr, "stencil3d gpu = %.3f ms, host = %.3f ms\n",
            acc_stencil.gpu_ms, acc_stencil.host_ms);

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms (%d timesteps/pass)\n",
            total_gpu_ms, num_timesteps);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_data);

    return 0;
}
