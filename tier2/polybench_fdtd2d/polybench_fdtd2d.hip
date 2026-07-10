// polybench_fdtd2d.hip — PolyBench FDTD-2D benchmark (HIP, self-contained)
//
// 2D Finite Difference Time Domain electromagnetic simulation.
// Three field arrays: ex[NX][NY], ey[NX][NY], hz[NX][NY]
//
// Per time step update equations (PolyBench standard):
//   ex[0][j]  = 0                                         (boundary)
//   ex[i][j] += 0.5*(hz[i][j] - hz[i-1][j])  for i >= 1
//   ey[i][0]  = 0                                         (boundary)
//   ey[i][j] += 0.5*(hz[i][j] - hz[i][j-1])  for j >= 1
//   hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j])
//                                              for i < NX-1, j < NY-1
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./polybench_fdtd2d [--size N] [--tmax T]
//
//   --size N        Grid dimension (N×N)            (default: 512)
//   --tmax T        Time steps per benchmark pass   (default: 50)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_tmax, BENCH_PARAM_block_size
//   BENCH_PARAM_warmup   (warmup passes, default 3)
//   BENCH_PARAM_iteration    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (fdtd_update_ex, fdtd_update_ey, fdtd_update_hz) carrying that
// kernel's accumulated GPU event time and host wall-clock time over the
// pass's TMAX time steps, then a summary whose total_time_ms is the SUM of
// all recorded GPU event times. Effective bandwidth is derived downstream
// from time_ms and params.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
// FDTD-2D kernels
// ---------------------------------------------------------------------------

// Kernel 1: Update ex
//   ex[0][j]  = 0                              for all j
//   ex[i][j] += 0.5*(hz[i][j] - hz[i-1][j])   for i >= 1, all j
__global__ void fdtd_update_ex(float* __restrict__ ex,
                                const float* __restrict__ hz,
                                int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    if (i == 0) {
        ex[0 * NY + j] = 0.0f;
    } else {
        ex[i * NY + j] += 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
    }
}

// Kernel 2: Update ey
//   ey[i][0]  = 0                              for all i
//   ey[i][j] += 0.5*(hz[i][j] - hz[i][j-1])   for all i, j >= 1
__global__ void fdtd_update_ey(float* __restrict__ ey,
                                const float* __restrict__ hz,
                                int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    if (j == 0) {
        ey[i * NY + 0] = 0.0f;
    } else {
        ey[i * NY + j] += 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
    }
}

// Kernel 3: Update hz
//   hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j])
//   for i < NX-1, j < NY-1
__global__ void fdtd_update_hz(const float* __restrict__ ex,
                                const float* __restrict__ ey,
                                float* __restrict__ hz,
                                int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX - 1 || j >= NY - 1) return;

    hz[i * NY + j] -= 0.7f * (ex[i * NY + (j + 1)] - ex[i * NY + j] +
                               ey[(i + 1) * NY + j] - ey[i * NY + j]);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int N    = parseIntParam(argc, argv, "--size", 512);
    int TMAX = parseIntParam(argc, argv, "--tmax",  50);
    int block_size = parseIntParam(argc, argv, "--block_size", 256);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_tmax");
    if (env_val) TMAX = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int NX = N, NY = N;
    size_t grid_bytes = (size_t)NX * NY * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d  |  TMAX: %d  |  Passes: %d warmup + %d timed\n\n",
            NX, NY, TMAX, num_warmup, num_iters);

    // Host allocations and initialization
    float* h_ex = (float*)malloc(grid_bytes);
    float* h_ey = (float*)malloc(grid_bytes);
    float* h_hz = (float*)malloc(grid_bytes);

    srand(42);
    for (int i = 0; i < NX * NY; i++) h_ex[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NX * NY; i++) h_ey[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < NX * NY; i++) h_hz[i] = (float)(rand() % 100) / 10.0f;

    // Device allocations
    float *d_ex, *d_ey, *d_hz;
    HIP_CHECK(hipMalloc(&d_ex, grid_bytes));
    HIP_CHECK(hipMalloc(&d_ey, grid_bytes));
    HIP_CHECK(hipMalloc(&d_hz, grid_bytes));

    // Kernel launch configuration
    dim3 block(16, 16);
    dim3 grid2d((NY + block.x - 1) / block.x, (NX + block.y - 1) / block.y);

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"tmax\":%d,\"block_size\":%d",
             N, TMAX, block_size);

    // Per-kernel accumulators: each pass runs TMAX time steps; the
    // accumulated GPU/host time of a kernel's TMAX launches becomes that
    // pass's record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_ex, acc_ey, acc_hz;

    // One full FDTD pass. The kernels update ex/ey/hz IN PLACE across the
    // TMAX steps, so the pristine fields are re-uploaded (untimed) at the
    // start of every pass to keep each pass's work identical.
    auto run_fdtd_pass = [&]() {
        bench_lap_reset(&acc_ex);
        bench_lap_reset(&acc_ey);
        bench_lap_reset(&acc_hz);

        HIP_CHECK(hipMemcpy(d_ex, h_ex, grid_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_ey, h_ey, grid_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_hz, h_hz, grid_bytes, hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int t = 0; t < TMAX; ++t) {
            bench_lap_add(&acc_ex, bench_time_op(&timer, [&]() {
                fdtd_update_ex<<<grid2d, block>>>(d_ex, d_hz, NX, NY);
            }));
            bench_lap_add(&acc_ey, bench_time_op(&timer, [&]() {
                fdtd_update_ey<<<grid2d, block>>>(d_ey, d_hz, NX, NY);
            }));
            bench_lap_add(&acc_hz, bench_time_op(&timer, [&]() {
                fdtd_update_hz<<<grid2d, block>>>(d_ex, d_ey, d_hz, NX, NY);
            }));
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_fdtd_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_fdtd_pass();
        bench_emit_iter("fdtd_update_ex", i, &acc_ex, params);
        bench_emit_iter("fdtd_update_ey", i, &acc_ey, params);
        bench_emit_iter("fdtd_update_hz", i, &acc_hz, params);
        total_gpu_ms += acc_ex.gpu_ms + acc_ey.gpu_ms + acc_hz.gpu_ms;
        fprintf(stderr, "pass %d: ex=%.3f ey=%.3f hz=%.3f ms\n",
                i, acc_ex.gpu_ms, acc_ey.gpu_ms, acc_hz.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes (%d steps/pass)\n",
            total_gpu_ms, num_iters, TMAX);

    HIP_CHECK(hipFree(d_ex));
    HIP_CHECK(hipFree(d_ey));
    HIP_CHECK(hipFree(d_hz));
    free(h_ex); free(h_ey); free(h_hz);

    return 0;
}
