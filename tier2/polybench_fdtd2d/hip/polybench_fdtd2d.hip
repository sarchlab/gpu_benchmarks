// polybench_fdtd2d.hip — PolyBench FDTD-2D benchmark (HIP, self-contained)
//
// 2D Finite Difference Time Domain electromagnetic simulation.
// Three field arrays: ex[NX][NY], ey[NX][NY], hz[NX][NY]
//
// Per time step update equations (canonical PolyBench FDTD-2D):
//   ey[0][j]  = _fict_[t]   with _fict_[t] = t              (time source row)
//   ey[i][j] -= 0.5*(hz[i][j] - hz[i-1][j])  for i >= 1
//   ex[i][j] -= 0.5*(hz[i][j] - hz[i][j-1])  for j >= 1
//   hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j])
//                                              for i < NX-1, j < NY-1
//
// This matches the canonical PolyBench FDTD-2D recurrence: ey carries the
// i-difference and the _fict_[t] source row, ex carries the j-difference, and
// the E-field updates subtract. (An earlier version had the ex/ey difference
// axes swapped, added instead of subtracted, and dropped the source.)
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
//   BENCH_PARAM_size, BENCH_PARAM_tmax
//
// Output (stdout): the TMAX-step simulation runs exactly once; one
// {"type":"kernel",...} JSON line per kernel (fdtd_update_ex,
// fdtd_update_ey, fdtd_update_hz) carrying that kernel's GPU event time and
// host wall-clock time accumulated over the TMAX time steps, then a summary
// whose total_time_ms is the SUM of all recorded GPU event times. Effective bandwidth is derived downstream
// from time_ms and params.

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "cpu_reference.h"

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

// Kernel 1: Update ex  (canonical PolyBench fdtd-2d)
//   ex[i][j] -= 0.5*(hz[i][j] - hz[i][j-1])   for j >= 1  (ex[i][0] unchanged)
__global__ void fdtd_update_ex(float* __restrict__ ex,
                                const float* __restrict__ hz,
                                int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    if (j >= 1) {
        ex[i * NY + j] -= 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
    }
}

// Kernel 2: Update ey  (canonical PolyBench fdtd-2d)
//   ey[0][j]  = _fict_[t]                       (time-varying source row)
//   ey[i][j] -= 0.5*(hz[i][j] - hz[i-1][j])     for i >= 1, all j
__global__ void fdtd_update_ey(float* __restrict__ ey,
                                const float* __restrict__ hz,
                                int NX, int NY, float fict_t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    if (i == 0) {
        ey[0 * NY + j] = fict_t;
    } else {
        ey[i * NY + j] -= 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
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

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_tmax");
    if (env_val) TMAX = atoi(env_val);

    if (N <= 0 || TMAX <= 0) {
        fprintf(stderr, "FDTD-2D requires positive size and tmax\n");
        return EXIT_FAILURE;
    }
    if (N > INT_MAX / N) {
        fprintf(stderr, "FDTD-2D size*size exceeds INT_MAX\n");
        return EXIT_FAILURE;
    }

    int NX = N, NY = N;
    size_t grid_bytes = (size_t)NX * NY * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d  |  TMAX: %d\n\n",
            NX, NY, TMAX);

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
             "\"size\":%d,\"tmax\":%d,\"verification\":\"full_cpu_v1\"",
             N, TMAX);

    // Per-kernel accumulators: the single pass runs TMAX time steps; the
    // accumulated GPU/host time of a kernel's TMAX launches becomes that
    // kernel's record.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_ex, acc_ey, acc_hz;

    // One full FDTD pass (run exactly once). The kernels update ex/ey/hz IN
    // PLACE across the TMAX steps, so the pristine fields are re-uploaded
    // (untimed) at the start of the pass to keep every execution identical.
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
                // _fict_[t] = (float)t, the PolyBench time-varying source.
                fdtd_update_ey<<<grid2d, block>>>(d_ey, d_hz, NX, NY, (float)t);
            }));
            bench_lap_add(&acc_hz, bench_time_op(&timer, [&]() {
                fdtd_update_hz<<<grid2d, block>>>(d_ex, d_ey, d_hz, NX, NY);
            }));
        }
    };

    run_fdtd_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("fdtd_update_ex", 0, &acc_ex, params);
    bench_emit_iter("fdtd_update_ey", 0, &acc_ey, params);
    bench_emit_iter("fdtd_update_hz", 0, &acc_hz, params);
    total_gpu_ms += acc_ex.gpu_ms + acc_ey.gpu_ms + acc_hz.gpu_ms;
    fprintf(stderr, "ex=%.3f ey=%.3f hz=%.3f ms\n",
            acc_ex.gpu_ms, acc_ey.gpu_ms, acc_hz.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms (%d steps/pass)\n",
            total_gpu_ms, TMAX);

    // Verify every measured final field after timing, without another GPU pass.
    const auto reference = cpu_fdtd(h_ex, h_ey, h_hz, NX, NY, TMAX);
    const std::vector<double>* expected[] = {&reference.ex, &reference.ey, &reference.hz};
    float* host[] = {h_ex, h_ey, h_hz};
    float* device[] = {d_ex, d_ey, d_hz};
    const char* names[] = {"ex", "ey", "hz"};
    size_t mismatches = 0, checked = 0;
    for (int stage = 0; stage < 3; ++stage) {
        // The complete independent reference has consumed all three inputs.
        HIP_CHECK(hipMemcpy(host[stage], device[stage], grid_bytes, hipMemcpyDeviceToHost));
        for (int i = 0; i < NX; ++i) for (int j = 0; j < NY; ++j) {
            const size_t q = (size_t)i * NY + j;
            const bool exact = stage == 0 ? j == 0 : stage == 1 ? i == 0
                               : (i == NX - 1 || j == NY - 1);
            const double value = (*expected[stage])[q];
            if (!fdtd_value_matches(host[stage][q], value, exact)) {
                if (mismatches < 10)
                    fprintf(stderr, "MISMATCH %s[%zu]: GPU=%.9g CPU=%.17g\n",
                            names[stage], q, host[stage][q], value);
                ++mismatches;
            }
            ++checked;
        }
    }
    fprintf(stderr, "FDTD verification: %s (%zu values, %zu mismatches)\n",
            mismatches ? "FAIL" : "PASS", checked, mismatches);

    HIP_CHECK(hipFree(d_ex));
    HIP_CHECK(hipFree(d_ey));
    HIP_CHECK(hipFree(d_hz));
    free(h_ex); free(h_ey); free(h_hz);

    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
