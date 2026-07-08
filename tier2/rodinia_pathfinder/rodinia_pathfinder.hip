// rodinia_pathfinder.hip — Rodinia PathFinder benchmark (HIP, self-contained)
//
// Computes the minimum-cost path through a 2D grid from top to bottom.
// Each row is processed by one GPU kernel launch (double-buffered).
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./rodinia_pathfinder [--rows R] [--cols C]
//
//   --rows R        Grid rows                  (default: 100000)
//   --cols C        Grid columns               (default: 100)
//
// Environment variables (override argv):
//   BENCH_PARAM_rows, BENCH_PARAM_cols, BENCH_PARAM_block_size
//   BENCH_PARAM_num_warmup   (warmup grid sweeps, default 3)
//   BENCH_PARAM_num_iters    (timed grid sweeps, each recorded, default 7)
//
// Output (stdout): per timed sweep, one {"type":"kernel",...} JSON line for
// dynproc_kernel carrying the GPU event time and host time of the whole
// rows-1 launch chain, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. Bandwidth (3*rows*cols*4 bytes / time) is
// derived downstream from time_ms and params.
// Output (stderr): Verification result and device info

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <algorithm>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// PathFinder kernel
//   - gpuWall: ROWS x COLS cost grid (row-major)
//   - gpuSrc:  COLS minimum-cost values from previous row
//   - gpuDst:  COLS minimum-cost values for current row (output)
//   - cols:    number of columns
//   - t:       current row index (1-based)
// ---------------------------------------------------------------------------
#define BLOCK_SIZE 256

__global__ void dynproc_kernel(const int* __restrict__ gpuWall,
                               const int* __restrict__ gpuSrc,
                               int* __restrict__       gpuDst,
                               int cols, int t) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    int left  = (col > 0)      ? gpuSrc[col - 1] : INT_MAX;
    int above = gpuSrc[col];
    int right = (col < cols-1) ? gpuSrc[col + 1] : INT_MAX;

    int min3 = min(min(left, above), right);
    gpuDst[col] = gpuWall[(long long)t * cols + col] + min3;
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------
static int parseIntParam(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
        }
    }
    return def;
}

// ---------------------------------------------------------------------------
// CPU reference for verification (small grid only)
// ---------------------------------------------------------------------------
static void cpu_pathfinder(const int* wall, int* result, int rows, int cols) {
    std::vector<int> src(cols), dst(cols);
    // Initialize with row 0
    for (int c = 0; c < cols; c++) src[c] = wall[c];
    // Sweep rows
    for (int t = 1; t < rows; t++) {
        for (int c = 0; c < cols; c++) {
            int left  = (c > 0)      ? src[c - 1] : INT_MAX;
            int above = src[c];
            int right = (c < cols-1) ? src[c + 1] : INT_MAX;
            dst[c] = wall[(long long)t * cols + c] + std::min({left, above, right});
        }
        std::swap(src, dst);
    }
    for (int c = 0; c < cols; c++) result[c] = src[c];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int cols  = 100;
    int rows  = 100000;
    int block_size_param = 256;

    // Command-line fallback
    rows  = parseIntParam(argc, argv, "--rows", rows);
    cols  = parseIntParam(argc, argv, "--cols", cols);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_cols");
    if (env_val) cols = atoi(env_val);
    env_val = getenv("BENCH_PARAM_rows");
    if (env_val) rows = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size_param = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d rows x %d cols\n\n",
            rows, cols);

    long long total_cells = (long long)rows * cols;
    size_t wall_bytes = (size_t)total_cells * sizeof(int);
    size_t row_bytes  = (size_t)cols * sizeof(int);

    // Host allocations
    int* h_wall = (int*)malloc(wall_bytes);
    int* h_src  = (int*)malloc(row_bytes);
    int* h_result = (int*)malloc(row_bytes);

    // Initialize wall with random values in [0, 10)
    srand(42);
    for (long long i = 0; i < total_cells; i++) {
        h_wall[i] = rand() % 10;
    }
    // Initialize src = row 0 costs
    for (int c = 0; c < cols; c++) {
        h_src[c] = h_wall[c];
    }

    // -----------------------------------------------------------------
    // Verification with small grid (avoid OOM for large grids)
    // -----------------------------------------------------------------
    {
        int vrows = (rows  < 1000) ? rows  : 1000;
        int vcols = (cols  < 100)  ? cols  : 100;
        long long vcells = (long long)vrows * vcols;

        int* v_wall = (int*)malloc((size_t)vcells * sizeof(int));
        srand(123);
        for (long long i = 0; i < vcells; i++) v_wall[i] = rand() % 10;

        // CPU reference
        std::vector<int> cpu_result(vcols);
        cpu_pathfinder(v_wall, cpu_result.data(), vrows, vcols);

        // GPU reference (small grid)
        int *gv_wall, *gv_src, *gv_dst;
        HIP_CHECK(hipMalloc(&gv_wall, (size_t)vcells * sizeof(int)));
        HIP_CHECK(hipMalloc(&gv_src,  (size_t)vcols  * sizeof(int)));
        HIP_CHECK(hipMalloc(&gv_dst,  (size_t)vcols  * sizeof(int)));
        HIP_CHECK(hipMemcpy(gv_wall, v_wall, (size_t)vcells * sizeof(int), hipMemcpyHostToDevice));

        // Initialize src = row 0
        HIP_CHECK(hipMemcpy(gv_src, v_wall, (size_t)vcols * sizeof(int), hipMemcpyHostToDevice));

        int* gsrc = gv_src;
        int* gdst = gv_dst;
        int nblocks = (vcols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int t = 1; t < vrows; t++) {
            dynproc_kernel<<<nblocks, BLOCK_SIZE>>>(gv_wall, gsrc, gdst, vcols, t);
            int* tmp = gsrc; gsrc = gdst; gdst = tmp;
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<int> gpu_result(vcols);
        HIP_CHECK(hipMemcpy(gpu_result.data(), gsrc, (size_t)vcols * sizeof(int), hipMemcpyDeviceToHost));

        bool ok = true;
        for (int c = 0; c < vcols && ok; c++) {
            if (cpu_result[c] != gpu_result[c]) {
                fprintf(stderr, "VERIFY FAIL at col %d: cpu=%d gpu=%d\n",
                        c, cpu_result[c], gpu_result[c]);
                ok = false;
            }
        }
        if (ok) fprintf(stderr, "Verification PASSED (small grid %dx%d)\n", vrows, vcols);

        HIP_CHECK(hipFree(gv_wall));
        HIP_CHECK(hipFree(gv_src));
        HIP_CHECK(hipFree(gv_dst));
        free(v_wall);
    }

    // -----------------------------------------------------------------
    // Device allocations for full benchmark
    // -----------------------------------------------------------------
    int *d_wall, *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_wall, wall_bytes));
    HIP_CHECK(hipMalloc(&d_src,  row_bytes));
    HIP_CHECK(hipMalloc(&d_dst,  row_bytes));
    HIP_CHECK(hipMemcpy(d_wall, h_wall, wall_bytes, hipMemcpyHostToDevice));

    int nblocks = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    char params[128];
    snprintf(params, sizeof(params),
             "\"cols\":%d,\"rows\":%d,\"block_size\":%d",
             cols, rows, block_size_param);

    // One full grid sweep. The sweep only reads d_wall (never modified) and
    // ping-pongs d_src/d_dst, so d_src is re-initialized to row 0 before
    // every sweep (untimed, as the old code did). The rows-1 dependent
    // kernel launches contain no interleaved transfers, only host-side
    // pointer swaps, so the whole chain is timed as ONE event bracket:
    // per-launch brackets (up to 100k launches) would serialize the launch
    // pipeline and distort the measurement.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t lap;
    int* final_src = d_src;

    auto run_sweep = [&]() {
        HIP_CHECK(hipMemcpy(d_src, h_src, row_bytes, hipMemcpyHostToDevice));
        int* src = d_src;
        int* dst = d_dst;
        lap = bench_time_op(&timer, [&]() {
            for (int t = 1; t < rows; t++) {
                dynproc_kernel<<<nblocks, BLOCK_SIZE>>>(d_wall, src, dst, cols, t);
                int* tmp = src; src = dst; dst = tmp;
            }
        });
        final_src = src;
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_sweep();
    }

    double total_gpu_ms = 0.0;
    for (int iter = 0; iter < num_iters; iter++) {
        run_sweep();
        bench_emit_iter("dynproc_kernel", iter, &lap, params);
        total_gpu_ms += lap.gpu_ms;
        fprintf(stderr, "dynproc_kernel iter %d: gpu = %.3f ms, host = %.3f ms\n",
                iter, lap.gpu_ms, lap.host_ms);
    }

    bench_timer_destroy(&timer);

    // Copy final-row result back from the last sweep
    HIP_CHECK(hipMemcpy(h_result, final_src, row_bytes, hipMemcpyDeviceToHost));

    // Find min of result row
    int min_cost = h_result[0];
    for (int c = 1; c < cols; c++) {
        if (h_result[c] < min_cost) min_cost = h_result[c];
    }
    fprintf(stderr, "Min cost in final row: %d\n", min_cost);

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d sweeps\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_wall));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    free(h_wall);
    free(h_src);
    free(h_result);

    return 0;
}
