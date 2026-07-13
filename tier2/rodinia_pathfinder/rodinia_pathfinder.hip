// rodinia_pathfinder.hip — Rodinia PathFinder benchmark (HIP, self-contained)
//
// Dynamic-programming shortest path down a 2D cost grid: starting from the top
// row, each row's minimum accumulated cost is derived from the three cells
// above it (left / up / right).
//
// This is a faithful port of the Rodinia 3.1 CUDA PathFinder: the device
// kernel (dynproc_kernel) and its pyramidal / ghost-zone temporal-blocking
// scheme are the ORIGINAL — each launch advances `pyramid_height` rows for a
// 1-D block loaded once into shared memory, with a border (halo) so
// neighbouring blocks stay consistent. The host driver (calc_path launch loop
// + pyramid geometry) is the original too; only the outer loop is restructured
// for the single-pass timing harness and JSON output.
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA) via the HIP compat layer.
//
// Usage:
//   ./rodinia_pathfinder [--cols C] [--rows R] [--pyramid_height P]
//
//   --cols C           Grid columns (width)              (default: 100000)
//   --rows R           Grid rows (height / #time steps)  (default: 100)
//   --pyramid_height P Rows advanced per kernel launch   (default: 20)
//
// Environment variables (override argv):
//   BENCH_PARAM_cols, BENCH_PARAM_rows, BENCH_PARAM_pyramid_height
//
// Output (stdout): one {"type":"kernel",...} JSON line for
// dynproc_kernel carrying the accumulated GPU event time of all pyramidal
// launches within the pass (a full top-to-bottom grid traversal) plus host
// wall-clock time, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times.
// Output (stderr): verification result and device info.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <algorithm>
#include <unistd.h>   // sysconf — physical-RAM check before the host malloc

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static int parseIntParam(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return def;
}

// ---------------------------------------------------------------------------
// Constants and kernel — verbatim from Rodinia 3.1 cuda/pathfinder/pathfinder.cu
// ---------------------------------------------------------------------------
#define BLOCK_SIZE 256
#define HALO 1  // halo width along one direction when advancing to next iteration

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration,
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols,
                int rows,
                int startStep,
                int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  // 64-bit index: cols*(startStep+i) overflows int once cols is
                  // large (e.g. cols=5,000,000 overflows at row ~430), causing
                  // an illegal access. The host reference already uses long long.
                  long long index = (long long)cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];

            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];
      }
}

// ---------------------------------------------------------------------------
// CPU reference for verification (same DP recurrence the pyramid computes)
// ---------------------------------------------------------------------------
static void cpu_pathfinder(const int* data, int* result, int rows, int cols) {
    std::vector<int> src(cols), dst(cols);
    for (int c = 0; c < cols; c++) src[c] = data[c];  // row 0
    for (int t = 0; t < rows - 1; t++) {
        for (int n = 0; n < cols; n++) {
            int minv = src[n];
            if (n > 0)        minv = std::min(minv, src[n - 1]);
            if (n < cols - 1) minv = std::min(minv, src[n + 1]);
            dst[n] = data[(long long)cols * (t + 1) + n] + minv;
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

    int cols           = parseIntParam(argc, argv, "--cols",           100000);
    int rows           = parseIntParam(argc, argv, "--rows",              100);
    int pyramid_height = parseIntParam(argc, argv, "--pyramid_height",     20);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_cols");           if (env_val) cols = atoi(env_val);
    env_val = getenv("BENCH_PARAM_rows");           if (env_val) rows = atoi(env_val);
    env_val = getenv("BENCH_PARAM_pyramid_height"); if (env_val) pyramid_height = atoi(env_val);

    // pyramid_height must keep the small-block size positive.
    if (pyramid_height * HALO * 2 >= BLOCK_SIZE) {
        fprintf(stderr, "pyramid_height too large for BLOCK_SIZE=%d (need pyramid_height < %d)\n",
                BLOCK_SIZE, BLOCK_SIZE / (2 * HALO));
        return EXIT_FAILURE;
    }

    // ------------------- pyramid parameters (original run()) -------------------
    int borderCols    = (pyramid_height) * HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height) * HALO * 2;
    int blockCols     = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "pyramidHeight: %d  gridSize(cols): %d  border: %d  BLOCK_SIZE: %d  blockGrid: %d  targetBlock: %d\n",
            pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
    fprintf(stderr, "Grid: %d rows x %d cols  (launches/pass = ceil((rows-1)/pyramid_height))\n\n",
            rows, cols);

    long long total_cells = (long long)rows * cols;
    long long grid_bytes  = total_cells * (long long)sizeof(int);

    // Preflight BOTH allocations before touching a single byte, so an oversized
    // config fails fast and cleanly instead of OOM-killing the process/runner.
    //
    // (1) Device memory is the binding limit. The device holds
    //     gpuWall + gpuResult[0] + gpuResult[1] = wall_bytes + 2*row_bytes, and
    //     the cold-cache warmer needs ~768 MB more. No real GPU has ~200 GB, so
    //     the 5,000,000 x 10,000 grid is impossible on-device regardless of host
    //     RAM — check this FIRST, before initializing ~200 GB of host memory
    //     only to fail at hipMalloc.
    long long row_bytes_ll  = (long long)cols * (long long)sizeof(int);
    long long wall_bytes_ll = (total_cells - cols) * (long long)sizeof(int);
    long long dev_needed    = wall_bytes_ll + 2 * row_bytes_ll;
    const long long warmer_reserve = 1LL << 30;  // ~1 GB for the warmer + overhead
    if (dev_needed + warmer_reserve > (long long)prop.totalGlobalMem) {
        fprintf(stderr,
                "ERROR: device buffers need %.1f GB but %s has %.1f GB "
                "(rows=%d x cols=%d). Reduce rows and/or cols.\n",
                (double)dev_needed / 1e9, prop.name,
                (double)prop.totalGlobalMem / 1e9, rows, cols);
        return EXIT_FAILURE;
    }

    // (2) Host cost grid `data`. Under Linux memory overcommit, malloc for a
    //     huge grid can return non-NULL and only OOM-kill the process once the
    //     init loop faults the pages in — so a post-malloc NULL check is not
    //     enough. Refuse if it would exceed HALF of physical RAM (the OS, other
    //     processes, and page tables also need room).
    long phys_pages = sysconf(_SC_PHYS_PAGES);
    long page_size  = sysconf(_SC_PAGE_SIZE);
    if (phys_pages > 0 && page_size > 0) {
        long long phys_bytes = (long long)phys_pages * (long long)page_size;
        if (grid_bytes > phys_bytes / 2) {
            fprintf(stderr,
                    "ERROR: cost grid needs %.1f GB, over half of the host's "
                    "%.1f GB RAM (rows=%d x cols=%d). Reduce rows and/or cols.\n",
                    (double)grid_bytes / 1e9, (double)phys_bytes / 1e9, rows, cols);
            return EXIT_FAILURE;
        }
    }

    // Host allocation + synthetic cost grid (Rodinia uses rand()%10, seed 9).
    int* data = (int*)malloc((size_t)grid_bytes);
    if (!data) {
        fprintf(stderr,
                "ERROR: failed to allocate %.1f GB host cost grid "
                "(rows=%d x cols=%d). Reduce rows and/or cols.\n",
                (double)grid_bytes / 1e9, rows, cols);
        return EXIT_FAILURE;
    }
    srand(9);
    for (long long i = 0; i < total_cells; i++) data[i] = rand() % 10;

    // Device buffers (original layout): gpuWall holds rows 1..rows-1;
    // gpuResult[0]/[1] are the ping-pong current/next row-cost vectors.
    int *gpuWall, *gpuResult[2];
    size_t row_bytes  = (size_t)cols * sizeof(int);
    size_t wall_bytes = (size_t)(total_cells - cols) * sizeof(int);
    HIP_CHECK(hipMalloc(&gpuResult[0], row_bytes));
    HIP_CHECK(hipMalloc(&gpuResult[1], row_bytes));
    HIP_CHECK(hipMalloc(&gpuWall,      wall_bytes));
    HIP_CHECK(hipMemcpy(gpuWall, data + cols, wall_bytes, hipMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    char params[160];
    snprintf(params, sizeof(params),
             "\"cols\":%d,\"rows\":%d,\"pyramid_height\":%d,\"block_size\":%d",
             cols, rows, pyramid_height, BLOCK_SIZE);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;
    int final_ret = 0;

    // One recorded pass = a full top-to-bottom traversal of all `rows`, advanced
    // pyramid_height rows per launch (original calc_path loop). Every pyramidal
    // launch is timed and accumulated into one dynproc_kernel record. gpuWall is
    // read-only; gpuResult[0] is re-seeded with row 0 (untimed) at the start of
    // every pass so each pass traverses the identical grid.
    auto run_pass = [&]() {
        bench_lap_reset(&acc);
        HIP_CHECK(hipMemcpy(gpuResult[0], data, row_bytes, hipMemcpyHostToDevice));
        int src = 1, dst = 0;
        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int t = 0; t < rows - 1; t += pyramid_height) {
            int temp = src; src = dst; dst = temp;
            int it = MIN(pyramid_height, rows - t - 1);
            bench_lap_add(&acc, bench_time_op(&timer, [&]() {
                dynproc_kernel<<<dimGrid, dimBlock>>>(
                    it, gpuWall, gpuResult[src], gpuResult[dst],
                    cols, rows, t, borderCols);
            }));
        }
        final_ret = dst;
    };

    double total_gpu_ms = 0.0;
    run_pass();
    bench_emit_iter("dynproc_kernel", 0, &acc, params);
    total_gpu_ms += acc.gpu_ms;
    fprintf(stderr, "dynproc_kernel gpu = %.3f ms, host = %.3f ms\n",
            acc.gpu_ms, acc.host_ms);

    bench_timer_destroy(&timer);

    // Verify the final row against the CPU DP reference.
    std::vector<int> gpu_result(cols), cpu_result(cols);
    HIP_CHECK(hipMemcpy(gpu_result.data(), gpuResult[final_ret], row_bytes, hipMemcpyDeviceToHost));
    cpu_pathfinder(data, cpu_result.data(), rows, cols);
    bool ok = true;
    for (int c = 0; c < cols && ok; c++) {
        if (cpu_result[c] != gpu_result[c]) {
            fprintf(stderr, "VERIFY FAIL at col %d: cpu=%d gpu=%d\n", c, cpu_result[c], gpu_result[c]);
            ok = false;
        }
    }
    if (ok) fprintf(stderr, "Verification PASSED (%d x %d grid)\n", rows, cols);

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    HIP_CHECK(hipFree(gpuWall));
    HIP_CHECK(hipFree(gpuResult[0]));
    HIP_CHECK(hipFree(gpuResult[1]));
    free(data);

    return ok ? 0 : 1;
}
