// rodinia_hotspot.hip — Rodinia Hotspot benchmark (HIP, self-contained)
//
// Iterative 2D stencil thermal simulation for chip temperature estimation.
// Each cell's temperature is updated based on its four neighbors (N/S/E/W),
// the local power density, and thermal resistances.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_hotspot [--grid_size N] [--block_size B] [--num_iterations K]
//
//   --grid_size N       Grid dimension (N×N)         (default: 512)
//   --block_size B      Thread-block dimension (B×B) (default: 16)
//   --num_iterations K  Stencil time-steps per timed launch (default: 10)
//
// Environment variables (override argv):
//   BENCH_PARAM_grid_size, BENCH_PARAM_block_size, BENCH_PARAM_num_iterations
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line for
// hotspot_kernel carrying the accumulated GPU event time of all stencil steps
// within the pass plus host wall-clock time, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Effective
// bandwidth is derived downstream from time_ms and params.
// Output (stderr): human-readable results

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
// Chip thermal constants (Rodinia defaults)
// ---------------------------------------------------------------------------
#define AMB_TEMP    80.0f
#define CHIP_HEIGHT 0.016f
#define CHIP_WIDTH  0.016f
#define T_CHIP      0.0005f
#define K_SI        100.0f
#define C_SI        1.75e6f

// ---------------------------------------------------------------------------
// Hotspot stencil kernel
// ---------------------------------------------------------------------------
__global__ void hotspot_kernel(const float* __restrict__ temp_src,
                                float* __restrict__ temp_dst,
                                const float* __restrict__ power,
                                int grid_cols, int grid_rows,
                                float step_div_cap,
                                float Rx_1, float Ry_1, float Rz_1) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < grid_cols && row < grid_rows) {
        int idx = row * grid_cols + col;

        float temp_c = temp_src[idx];
        float temp_n = (row > 0)             ? temp_src[(row - 1) * grid_cols + col] : temp_c;
        float temp_s = (row < grid_rows - 1) ? temp_src[(row + 1) * grid_cols + col] : temp_c;
        float temp_w = (col > 0)             ? temp_src[row * grid_cols + (col - 1)] : temp_c;
        float temp_e = (col < grid_cols - 1) ? temp_src[row * grid_cols + (col + 1)] : temp_c;

        float delta = step_div_cap *
            (power[idx]
             + (temp_n + temp_s - 2.0f * temp_c) * Ry_1
             + (temp_w + temp_e - 2.0f * temp_c) * Rx_1
             + (AMB_TEMP - temp_c) * Rz_1);

        temp_dst[idx] = temp_c + delta;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int grid_size      = parseIntParam(argc, argv, "--grid_size",      512);
    int block_dim      = parseIntParam(argc, argv, "--block_size",      16);
    int num_iterations = parseIntParam(argc, argv, "--num_iterations",  10);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_grid_size");
    if (env_val) grid_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_dim = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_iterations");
    if (env_val) num_iterations = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int grid_rows  = grid_size;
    int grid_cols  = grid_size;
    int total_cells = grid_rows * grid_cols;

    // Compute thermal parameters
    float grid_height = CHIP_HEIGHT / grid_rows;
    float grid_width  = CHIP_WIDTH  / grid_cols;
    float cap         = C_SI * T_CHIP * grid_height * grid_width;
    float Rx          = grid_width  / (2.0f * K_SI * T_CHIP * grid_height);
    float Ry          = grid_height / (2.0f * K_SI * T_CHIP * grid_width);
    float Rz          = T_CHIP / (K_SI * grid_height * grid_width);
    float max_slope   = K_SI / (0.5f * T_CHIP * C_SI);
    float step        = 0.001f / max_slope;
    float step_div_cap = step / cap;
    float Rx_1        = 1.0f / Rx;
    float Ry_1        = 1.0f / Ry;
    float Rz_1        = 1.0f / Rz;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d  |  Block: %d×%d  |  Steps/launch: %d\n\n",
            grid_rows, grid_cols, block_dim, block_dim, num_iterations);

    // Host allocations
    float* h_temp  = (float*)malloc(total_cells * sizeof(float));
    float* h_power = (float*)malloc(total_cells * sizeof(float));

    // Generate synthetic temperature and power grids
    srand(42);
    for (int i = 0; i < total_cells; ++i) {
        h_temp[i]  = AMB_TEMP + (float)(rand() % 200) / 10.0f;
        h_power[i] = (float)(rand() % 100) / 500.0f;
    }

    // Device allocations
    float *d_temp_src, *d_temp_dst, *d_power;
    HIP_CHECK(hipMalloc(&d_temp_src, total_cells * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_temp_dst, total_cells * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_power,    total_cells * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_power, h_power, total_cells * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(block_dim, block_dim);
    dim3 grid((grid_cols + block_dim - 1) / block_dim,
              (grid_rows + block_dim - 1) / block_dim);

    char params[160];
    snprintf(params, sizeof(params),
             "\"grid_size\":%d,\"block_size\":%d,\"num_iterations\":%d",
             grid_size, block_dim, num_iterations);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_hotspot;

    // One full pass = num_iterations ping-pong stencil steps. The initial
    // temperature grid is re-uploaded at the start of every pass (untimed),
    // as the pre-conversion code did, so each pass simulates the same
    // trajectory regardless of what earlier passes left in the buffers.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_hotspot);

        HIP_CHECK(hipMemcpy(d_temp_src, h_temp,
                            total_cells * sizeof(float), hipMemcpyHostToDevice));

        float* src = d_temp_src;
        float* dst = d_temp_dst;

        for (int k = 0; k < num_iterations; ++k) {
            bench_lap_add(&acc_hotspot, bench_time_op(&timer, [&]() {
                hotspot_kernel<<<grid, block>>>(
                    src, dst, d_power, grid_cols, grid_rows, step_div_cap, Rx_1, Ry_1, Rz_1);
            }));
            // Ping-pong buffers
            float* tmp = src;
            src = dst;
            dst = tmp;
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_pass();
        bench_emit_iter("hotspot_kernel", i, &acc_hotspot, params);
        total_gpu_ms += acc_hotspot.gpu_ms;
        fprintf(stderr, "pass %d: hotspot_kernel gpu = %.3f ms, host = %.3f ms\n",
                i, acc_hotspot.gpu_ms, acc_hotspot.host_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes (%d steps/pass)\n",
            total_gpu_ms, num_iters, num_iterations);

    HIP_CHECK(hipFree(d_temp_src));
    HIP_CHECK(hipFree(d_temp_dst));
    HIP_CHECK(hipFree(d_power));
    free(h_temp);
    free(h_power);

    return 0;
}
