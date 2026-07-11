// rodinia_hotspot.hip — Rodinia Hotspot benchmark (HIP, self-contained)
//
// Transient thermal simulation of a chip: an iterative 2D structured-grid
// stencil where each cell's temperature is updated from its four neighbours,
// the local power density, and thermal resistances.
//
// This is a faithful port of the Rodinia 3.1 CUDA Hotspot: the device kernel
// (calculate_temp) and its pyramidal / ghost-zone temporal-blocking scheme
// are the original — each launch advances `pyramid_height` time steps for a
// block loaded once into shared memory, with a halo (border) so neighbouring
// blocks stay consistent. Only the host driver is restructured for the
// single-pass timing harness and JSON output.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_hotspot [--grid_size N] [--total_iterations T] [--pyramid_height P]
//
//   --grid_size N        Grid dimension (N×N)               (default: 1024)
//   --total_iterations T Total thermal time steps simulated (default: 360)
//   --pyramid_height P    Time steps advanced per launch     (default: 2)
//
// Environment variables (override argv):
//   BENCH_PARAM_grid_size, BENCH_PARAM_total_iterations, BENCH_PARAM_pyramid_height
//
// Output (stdout): one {"type":"kernel",...} JSON line for
// calculate_temp carrying the accumulated GPU event time of all pyramidal
// launches within the pass plus host wall-clock time, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times.
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
// Constants and kernel — verbatim from Rodinia 3.1 cuda/hotspot/hotspot.cu
// ---------------------------------------------------------------------------
#define BLOCK_SIZE      16
#define MAX_PD          (3.0e6)
#define PRECISION       0.001
#define SPEC_HEAT_SI    1.75e6
#define K_SI            100
#define FACTOR_CHIP     0.5
#define EXPAND_RATE     2  // add one iteration extends the pyramid base by 2 per border

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

/* chip parameters */
static const float t_chip     = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp   = 80.0f;

__global__ void calculate_temp(int iteration,   // number of iterations
                               float *power,     // power input
                               float *temp_src,  // temperature input/output
                               float *temp_dst,  // temperature input/output
                               int grid_cols,    // Col of grid
                               int grid_rows,    // Row of grid
                               int border_cols,  // border offset
                               int border_rows,  // border offset
                               float Cap,         // Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed) {

    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result

    float amb_temp = 80.0f;
    float step_div_Cap;
    float Rx_1, Ry_1, Rz_1;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    step_div_Cap = step / Cap;

    Rx_1 = 1 / Rx;
    Ry_1 = 1 / Ry;
    Rz_1 = 1 / Rz;

    // each block finally computes result for a small block after N iterations.
    // it is the non-overlapping small blocks that cover all the input data

    // calculate the small block size
    int small_block_rows = BLOCK_SIZE - iteration * 2; // EXPAND_RATE
    int small_block_cols = BLOCK_SIZE - iteration * 2; // EXPAND_RATE

    // calculate the boundary for the block according to the boundary of its small block
    int blkY = small_block_rows * by - border_rows;
    int blkX = small_block_cols * bx - border_cols;
    int blkYmax = blkY + BLOCK_SIZE - 1;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int yidx = blkY + ty;
    int xidx = blkX + tx;

    // load data if it is within the valid input range
    int loadYidx = yidx, loadXidx = xidx;
    int index = grid_cols * loadYidx + loadXidx;

    if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
        temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
        power_on_cuda[ty][tx] = power[index];    // Load the power data from global memory to shared memory
    }
    __syncthreads();

    // effective range within this block that falls within the valid range of the
    // input data used to rule out computation outside the boundary.
    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

    int N = ty - 1;
    int S = ty + 1;
    int W = tx - 1;
    int E = tx + 1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(tx, validXmin, validXmax) &&
            IN_RANGE(ty, validYmin, validYmax)) {
            computed = true;
            temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
                (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
                (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
        }
        __syncthreads();
        if (i == iteration - 1)
            break;
        if (computed) // Assign the computation range
            temp_on_cuda[ty][tx] = temp_t[ty][tx];
        __syncthreads();
    }

    // update the global memory: after the last iteration, only threads
    // coordinated within the small block perform the calculation.
    if (computed) {
        temp_dst[index] = temp_t[ty][tx];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int grid_size        = parseIntParam(argc, argv, "--grid_size",        1024);
    int total_iterations = parseIntParam(argc, argv, "--total_iterations",  360);
    int pyramid_height   = parseIntParam(argc, argv, "--pyramid_height",      2);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_grid_size");        if (env_val) grid_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_total_iterations"); if (env_val) total_iterations = atoi(env_val);
    env_val = getenv("BENCH_PARAM_pyramid_height");   if (env_val) pyramid_height = atoi(env_val);

    // pyramid_height must keep the small-block size positive.
    if (pyramid_height * EXPAND_RATE >= BLOCK_SIZE) {
        fprintf(stderr, "pyramid_height too large for BLOCK_SIZE=%d (need pyramid_height < %d)\n",
                BLOCK_SIZE, BLOCK_SIZE / EXPAND_RATE);
        return EXIT_FAILURE;
    }

    int grid_rows = grid_size;
    int grid_cols = grid_size;
    int total_cells = grid_rows * grid_cols;

    // Pyramid geometry (original compute_tran_temp setup)
    int borderCols = (pyramid_height) * EXPAND_RATE / 2;
    int borderRows = (pyramid_height) * EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
    int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

    // Thermal coefficients (original compute_tran_temp)
    float grid_height = chip_height / grid_rows;
    float grid_width  = chip_width  / grid_cols;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx  = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    float Ry  = grid_height / (2.0f * K_SI * t_chip * grid_width);
    float Rz  = t_chip / (K_SI * grid_height * grid_width);
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float time_elapsed = 0.001f;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d  |  pyramid_height: %d  |  total_iterations: %d\n",
            grid_rows, grid_cols, pyramid_height, total_iterations);
    fprintf(stderr, "Grid of blocks: %d×%d (BLOCK_SIZE=%d)\n\n", blockCols, blockRows, BLOCK_SIZE);

    // Host allocations + synthetic temperature / power grids
    float* h_temp  = (float*)malloc(total_cells * sizeof(float));
    float* h_power = (float*)malloc(total_cells * sizeof(float));
    srand(42);
    for (int i = 0; i < total_cells; ++i) {
        h_temp[i]  = amb_temp + (float)(rand() % 200) / 10.0f;
        h_power[i] = (float)(rand() % 100) / 500.0f;
    }

    float* MatrixTemp[2];
    float* d_power;
    HIP_CHECK(hipMalloc(&MatrixTemp[0], total_cells * sizeof(float)));
    HIP_CHECK(hipMalloc(&MatrixTemp[1], total_cells * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_power,       total_cells * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_power, h_power, total_cells * sizeof(float), hipMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);

    char params[192];
    snprintf(params, sizeof(params),
             "\"grid_size\":%d,\"total_iterations\":%d,\"pyramid_height\":%d,\"block_size\":%d",
             grid_size, total_iterations, pyramid_height, BLOCK_SIZE);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;

    // One recorded pass = a full transient simulation of total_iterations time
    // steps, advanced pyramid_height steps per launch (original compute_tran_temp
    // loop). Every pyramidal launch is timed and accumulated into one
    // calculate_temp record. The temperature grid is re-uploaded (untimed) at
    // the start of each pass so every pass simulates the identical trajectory.
    auto run_pass = [&]() {
        bench_lap_reset(&acc);
        HIP_CHECK(hipMemcpy(MatrixTemp[0], h_temp,
                            total_cells * sizeof(float), hipMemcpyHostToDevice));
        int src = 1, dst = 0;
        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int t = 0; t < total_iterations; t += pyramid_height) {
            int tmp = src; src = dst; dst = tmp;
            int it = MIN(pyramid_height, total_iterations - t);
            bench_lap_add(&acc, bench_time_op(&timer, [&]() {
                calculate_temp<<<dimGrid, dimBlock>>>(
                    it, d_power, MatrixTemp[src], MatrixTemp[dst],
                    grid_cols, grid_rows, borderCols, borderRows,
                    Cap, Rx, Ry, Rz, step, time_elapsed);
            }));
        }
    };

    run_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("calculate_temp", 0, &acc, params);
    total_gpu_ms += acc.gpu_ms;
    fprintf(stderr, "calculate_temp gpu = %.3f ms, host = %.3f ms\n",
            acc.gpu_ms, acc.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over 1 pass (%d steps/pass)\n",
            total_gpu_ms, total_iterations);

    HIP_CHECK(hipFree(MatrixTemp[0]));
    HIP_CHECK(hipFree(MatrixTemp[1]));
    HIP_CHECK(hipFree(d_power));
    free(h_temp);
    free(h_power);

    return 0;
}
