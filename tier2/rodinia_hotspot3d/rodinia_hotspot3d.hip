// rodinia_hotspot3d.hip — Rodinia HotSpot3D benchmark (HIP, self-contained)
//
// 3D stencil thermal simulation extending the 2D HotSpot benchmark.
// Each cell's temperature is updated based on its six neighbors (±x,±y,±z),
// local power density, and thermal resistances in an NxNxN grid.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables (override argv):
//   BENCH_PARAM_grid_size, BENCH_PARAM_block_size,
//   BENCH_PARAM_num_iterations, BENCH_PARAM_amb_temp
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line
// carrying the kernel's accumulated GPU event time and host wall-clock time
// over the pass's num_iterations stencil steps, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Effective
// bandwidth is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

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

static float parseFloatParam(int argc, char** argv, const char* name, float defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            float v = (float)atof(argv[i + 1]);
            if (v > 0.0f) return v;
            break;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Chip thermal constants (Rodinia defaults, extended to 3D)
// ---------------------------------------------------------------------------
#define CHIP_HEIGHT 0.016f
#define CHIP_WIDTH  0.016f
#define CHIP_DEPTH  0.016f
#define T_CHIP      0.0005f
#define K_SI        100.0f
#define C_SI        1.75e6f

// ---------------------------------------------------------------------------
// 3D Hotspot stencil kernel
// ---------------------------------------------------------------------------
__global__ void hotspot3d_kernel(const float* __restrict__ temp_src,
                                  float* __restrict__ temp_dst,
                                  const float* __restrict__ power,
                                  int nx, int ny, int nz,
                                  float step_div_cap,
                                  float Rx_1, float Ry_1, float Rz_1,
                                  float Ra_1, float amb_temp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = z * ny * nx + y * nx + x;

    float tc = temp_src[idx];

    // ±x neighbors (clamped boundary)
    float txm = (x > 0)      ? temp_src[z * ny * nx + y * nx + (x - 1)] : tc;
    float txp = (x < nx - 1) ? temp_src[z * ny * nx + y * nx + (x + 1)] : tc;

    // ±y neighbors (clamped boundary)
    float tym = (y > 0)      ? temp_src[z * ny * nx + (y - 1) * nx + x] : tc;
    float typ = (y < ny - 1) ? temp_src[z * ny * nx + (y + 1) * nx + x] : tc;

    // ±z neighbors (clamped boundary)
    float tzm = (z > 0)      ? temp_src[(z - 1) * ny * nx + y * nx + x] : tc;
    float tzp = (z < nz - 1) ? temp_src[(z + 1) * ny * nx + y * nx + x] : tc;

    float delta = step_div_cap * (
        power[idx]
        + (txm + txp - 2.0f * tc) * Rx_1
        + (tym + typ - 2.0f * tc) * Ry_1
        + (tzm + tzp - 2.0f * tc) * Rz_1
        + (amb_temp - tc) * Ra_1
    );

    temp_dst[idx] = tc + delta;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int grid_size      = 64;
    int block_dim      = 8;
    int num_iterations = 10;
    float amb_temp     = 80.0f;

    // Command-line fallback
    grid_size      = parseIntParam(argc, argv, "--grid_size", grid_size);
    block_dim      = parseIntParam(argc, argv, "--block_size", block_dim);
    num_iterations = parseIntParam(argc, argv, "--num_iterations", num_iterations);
    amb_temp       = parseFloatParam(argc, argv, "--amb_temp", amb_temp);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_grid_size");
    if (env_val) grid_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_dim = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_iterations");
    if (env_val) num_iterations = atoi(env_val);
    env_val = getenv("BENCH_PARAM_amb_temp");
    if (env_val) amb_temp = (float)atof(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int nx = grid_size, ny = grid_size, nz = grid_size;
    long long total_cells = (long long)nx * ny * nz;

    // Compute thermal parameters for 3D
    float dx = CHIP_WIDTH  / nx;
    float dy = CHIP_HEIGHT / ny;
    float dz = CHIP_DEPTH  / nz;

    float cap       = C_SI * T_CHIP * dx * dy;
    float Rx        = dx / (2.0f * K_SI * T_CHIP * dy);
    float Ry        = dy / (2.0f * K_SI * T_CHIP * dx);
    float Rz        = dz / (2.0f * K_SI * T_CHIP * dx);
    float Ra        = T_CHIP / (K_SI * dx * dy);
    float max_slope = K_SI / (0.5f * T_CHIP * C_SI);
    float step      = 0.001f / max_slope;
    float step_div_cap = step / cap;
    float Rx_1      = 1.0f / Rx;
    float Ry_1      = 1.0f / Ry;
    float Rz_1      = 1.0f / Rz;
    float Ra_1      = 1.0f / Ra;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d×%d  |  Block: %d³  |  Steps/pass: %d  |  "
            "Passes: %d warmup + %d timed\n\n",
            nx, ny, nz, block_dim, num_iterations, num_warmup, num_iters);

    // Host allocations
    size_t bytes = (size_t)total_cells * sizeof(float);
    float* h_temp  = (float*)malloc(bytes);
    float* h_power = (float*)malloc(bytes);

    // Generate synthetic temperature and power grids
    srand(42);
    for (long long i = 0; i < total_cells; ++i) {
        h_temp[i]  = amb_temp + (float)(rand() % 200) / 10.0f;
        h_power[i] = (float)(rand() % 100) / 500.0f;
    }

    // Device allocations
    float *d_temp_src, *d_temp_dst, *d_power;
    HIP_CHECK(hipMalloc(&d_temp_src, bytes));
    HIP_CHECK(hipMalloc(&d_temp_dst, bytes));
    HIP_CHECK(hipMalloc(&d_power,    bytes));

    HIP_CHECK(hipMemcpy(d_power, h_power, bytes, hipMemcpyHostToDevice));

    dim3 block(block_dim, block_dim, block_dim);
    dim3 grid_dim_val((nx + block_dim - 1) / block_dim,
                      (ny + block_dim - 1) / block_dim,
                      (nz + block_dim - 1) / block_dim);

    char params[160];
    snprintf(params, sizeof(params),
             "\"grid_size\":%d,\"block_size\":%d,\"num_iterations\":%d,\"amb_temp\":%.1f",
             grid_size, block_dim, num_iterations, amb_temp);

    // Per-pass accumulator: each pass runs num_iterations stencil steps with
    // ping-pong buffers; the accumulated GPU/host time becomes that pass's
    // record.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_hotspot;

    // One full HotSpot3D pass. The temperature grid evolves across steps, so
    // the pristine grid is re-uploaded (untimed) at the start of every pass
    // to keep each pass's work identical.
    auto run_hotspot_pass = [&]() {
        bench_lap_reset(&acc_hotspot);

        HIP_CHECK(hipMemcpy(d_temp_src, h_temp, bytes, hipMemcpyHostToDevice));
        float* src = d_temp_src;
        float* dst = d_temp_dst;
        for (int k = 0; k < num_iterations; ++k) {
            bench_lap_add(&acc_hotspot, bench_time_op(&timer, [&]() {
                hotspot3d_kernel<<<grid_dim_val, block>>>(
                    src, dst, d_power, nx, ny, nz, step_div_cap,
                    Rx_1, Ry_1, Rz_1, Ra_1, amb_temp);
            }));
            float* tmp = src; src = dst; dst = tmp;
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_hotspot_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_hotspot_pass();
        bench_emit_iter("hotspot3d_kernel", i, &acc_hotspot, params);
        total_gpu_ms += acc_hotspot.gpu_ms;
        fprintf(stderr, "pass %d: hotspot3d_kernel gpu = %.3f ms, host = %.3f ms\n",
                i, acc_hotspot.gpu_ms, acc_hotspot.host_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes (%d steps/pass)\n",
            total_gpu_ms, num_iters, num_iterations);

    // Cleanup
    HIP_CHECK(hipFree(d_temp_src));
    HIP_CHECK(hipFree(d_temp_dst));
    HIP_CHECK(hipFree(d_power));
    free(h_temp);
    free(h_power);

    return 0;
}
