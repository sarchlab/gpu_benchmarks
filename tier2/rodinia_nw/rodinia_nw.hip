// rodinia_nw.hip — Rodinia Needleman-Wunsch benchmark (HIP, self-contained)
//
// Dynamic programming sequence alignment using the Needleman-Wunsch algorithm.
// The NxN scoring matrix is filled using block-based anti-diagonal wavefront
// parallelism with shared memory. Two kernels cover the upper-left and
// lower-right triangles of the block dependency graph.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_nw [--sequence_length N] [--block_size B] [--penalty P]
//
//   --sequence_length N  Sequence length (scoring matrix is NxN) (default: 2048)
//   --block_size B       Block size (threads per block)          (default: 16)
//   --penalty P          Gap penalty                             (default: 2)
//
// Environment variables (override argv):
//   BENCH_PARAM_sequence_length, BENCH_PARAM_block_size, BENCH_PARAM_penalty
//   BENCH_PARAM_warmup   (warmup passes, default 3)
//   BENCH_PARAM_iteration    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (nw_kernel1, nw_kernel2) carrying that kernel's accumulated GPU
// event time and host time within the pass (summed over the wavefront
// diagonals), then a summary whose total_time_ms is the SUM of all recorded
// GPU event times. GCUPS / GB/s are derived downstream from time_ms + params.
// Output (stderr): device info, per-pass timing

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Inlined utilities
// ---------------------------------------------------------------------------

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
// NW scoring constants
// ---------------------------------------------------------------------------
#define MATCH_SCORE    1
#define MISMATCH_SCORE (-1)

__device__ static inline int nw_score(int a, int b) {
    return (a == b) ? MATCH_SCORE : MISMATCH_SCORE;
}

// ---------------------------------------------------------------------------
// Kernel 1: upper-left triangle of the block dependency graph
//
// For diagonal block_idx = 0..num_blocks-1, dispatches (block_idx+1) blocks.
// Block bx handles block position (row = block_idx - bx, col = bx).
// Each block uses shared memory to fill its (block_size x block_size) tile.
// ---------------------------------------------------------------------------
__global__ void nw_kernel1(int* __restrict__ d_seq,
                            int* __restrict__ d_matrix,
                            int cols,
                            int penalty,
                            int block_idx,
                            int block_size)
{
    int bx = hipBlockIdx_x;
    int tx = hipThreadIdx_x;

    // Filter out blocks beyond diagonal
    if (bx > block_idx) return;

    int b_row = block_idx - bx;
    int b_col = bx;

    // Global starting position (1-indexed within the matrix)
    int begin_row = b_row * block_size + 1;
    int begin_col = b_col * block_size + 1;

    // Shared memory layout:
    //   temp[0..(block_size+1)^2-1] : local DP tile (block_size+1 x block_size+1)
    //   ref_shared[0..block_size-1] : column sequence slice
    extern __shared__ int shared[];
    int* temp       = shared;
    int* ref_shared = &shared[(block_size + 1) * (block_size + 1)];

    // Load column sequence into shared memory
    if (tx < block_size) {
        ref_shared[tx] = d_seq[begin_col + tx];
    }
    __syncthreads();

    // Thread 0 loads the border row and column from global memory
    if (tx == 0) {
        // Left border column (from global matrix)
        for (int i = 0; i <= block_size; i++) {
            temp[i * (block_size + 1)] =
                d_matrix[(begin_row + i - 1) * cols + (begin_col - 1)];
        }
        // Top border row (from global matrix)
        for (int j = 0; j <= block_size; j++) {
            temp[j] = d_matrix[(begin_row - 1) * cols + (begin_col + j - 1)];
        }
    }
    __syncthreads();

    // Anti-diagonal sweep within the tile
    for (int diag = 0; diag < 2 * block_size - 1; diag++) {
        // Determine this thread's (i, j) within the tile [1..block_size]
        int i = (tx <= diag && tx < block_size && (diag - tx) < block_size)
                    ? tx + 1 : -1;
        int j = (i > 0) ? diag - tx + 1 : -1;

        if (i > 0 && j > 0 && i <= block_size && j <= block_size) {
            int idx_diag   = i * (block_size + 1) + j;
            int idx_up     = (i - 1) * (block_size + 1) + j;
            int idx_left   = i * (block_size + 1) + (j - 1);
            int idx_upleft = (i - 1) * (block_size + 1) + (j - 1);

            // Row sequence character is loaded directly from global memory
            int match = temp[idx_upleft] +
                        nw_score(d_seq[begin_row + i - 1], ref_shared[j - 1]);
            int del   = temp[idx_up]   - penalty;
            int ins   = temp[idx_left] - penalty;

            int val = match;
            if (del > val) val = del;
            if (ins > val) val = ins;
            temp[idx_diag] = val;
        }
        __syncthreads();
    }

    // Write computed tile back to global matrix
    if (tx < block_size) {
        for (int i = 1; i <= block_size; i++) {
            d_matrix[(begin_row + i - 1) * cols + (begin_col + tx)] =
                temp[i * (block_size + 1) + tx + 1];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: lower-right triangle of the block dependency graph
//
// For diagonal block_idx = 0..num_blocks-2, dispatches (num_blocks-1-block_idx) blocks.
// Block bx handles block position (row = num_blocks-1-bx, col = block_idx+1+bx).
// ---------------------------------------------------------------------------
__global__ void nw_kernel2(int* __restrict__ d_seq,
                            int* __restrict__ d_matrix,
                            int cols,
                            int penalty,
                            int block_idx,
                            int num_blocks,
                            int block_size)
{
    int bx = hipBlockIdx_x;
    int tx = hipThreadIdx_x;

    int blocks_on_diag = num_blocks - 1 - block_idx;
    if (bx >= blocks_on_diag) return;

    int b_row = num_blocks - 1 - bx;
    int b_col = block_idx + 1 + bx;

    int begin_row = b_row * block_size + 1;
    int begin_col = b_col * block_size + 1;

    extern __shared__ int shared[];
    int* temp       = shared;
    int* ref_shared = &shared[(block_size + 1) * (block_size + 1)];

    if (tx < block_size) {
        ref_shared[tx] = d_seq[begin_col + tx];
    }
    __syncthreads();

    if (tx == 0) {
        for (int i = 0; i <= block_size; i++) {
            temp[i * (block_size + 1)] =
                d_matrix[(begin_row + i - 1) * cols + (begin_col - 1)];
        }
        for (int j = 0; j <= block_size; j++) {
            temp[j] = d_matrix[(begin_row - 1) * cols + (begin_col + j - 1)];
        }
    }
    __syncthreads();

    for (int diag = 0; diag < 2 * block_size - 1; diag++) {
        int i = (tx <= diag && tx < block_size && (diag - tx) < block_size)
                    ? tx + 1 : -1;
        int j = (i > 0) ? diag - tx + 1 : -1;

        if (i > 0 && j > 0 && i <= block_size && j <= block_size) {
            int idx_diag   = i * (block_size + 1) + j;
            int idx_up     = (i - 1) * (block_size + 1) + j;
            int idx_left   = i * (block_size + 1) + (j - 1);
            int idx_upleft = (i - 1) * (block_size + 1) + (j - 1);

            int match = temp[idx_upleft] +
                        nw_score(d_seq[begin_row + i - 1], ref_shared[j - 1]);
            int del   = temp[idx_up]   - penalty;
            int ins   = temp[idx_left] - penalty;

            int val = match;
            if (del > val) val = del;
            if (ins > val) val = ins;
            temp[idx_diag] = val;
        }
        __syncthreads();
    }

    if (tx < block_size) {
        for (int i = 1; i <= block_size; i++) {
            d_matrix[(begin_row + i - 1) * cols + (begin_col + tx)] =
                temp[i * (block_size + 1) + tx + 1];
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int seq_len    = 2048;
    int block_size = 16;
    int penalty    = 2;

    // Command-line fallback
    seq_len    = parseIntParam(argc, argv, "--sequence_length", seq_len);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);
    penalty    = parseIntParam(argc, argv, "--penalty", penalty);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_sequence_length");
    if (env_val) seq_len = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_penalty");
    if (env_val) penalty = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Round sequence length up to a multiple of block_size
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int padded_len = num_blocks * block_size;

    // Matrix is (padded_len+1) x (padded_len+1); row 0 and col 0 are borders
    int cols = padded_len + 1;
    int rows = padded_len + 1;
    long matrix_elems = (long)rows * cols;

    // Allocate host arrays
    int* h_seq    = (int*)malloc((padded_len + 1) * sizeof(int));
    int* h_matrix = (int*)malloc(matrix_elems * sizeof(int));

    // Generate synthetic sequences (alphabet size 10, reproducible)
    srand(42);
    for (int i = 0; i <= padded_len; i++) {
        h_seq[i] = rand() % 10;
    }

    // Initialize DP border: row 0 and column 0
    for (int i = 0; i < rows; i++) {
        h_matrix[i * cols + 0] = -i * penalty;
    }
    for (int j = 0; j < cols; j++) {
        h_matrix[0 * cols + j] = -j * penalty;
    }

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Sequence length: %d  |  Block size: %d  |  Penalty: %d\n\n",
            seq_len, block_size, penalty);

    // Device allocations
    int *d_seq, *d_matrix;
    HIP_CHECK(hipMalloc(&d_seq,    (padded_len + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_matrix, matrix_elems * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_seq, h_seq, (padded_len + 1) * sizeof(int), hipMemcpyHostToDevice));

    // Shared memory per block: tile + ref slice
    size_t shared_mem = (size_t)(block_size + 1) * (block_size + 1) * sizeof(int)
                      + (size_t)block_size * sizeof(int);

    char params[128];
    snprintf(params, sizeof(params),
             "\"sequence_length\":%d,\"block_size\":%d,\"penalty\":%d",
             seq_len, block_size, penalty);

    // Per-kernel accumulators: each pass fills the whole DP matrix; the
    // accumulated GPU/host time of all wavefront diagonals within one pass
    // becomes that pass's record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2;

    // One full NW pass. The DP matrix (with its borders) is re-uploaded at
    // the start of every pass (untimed): the kernels fill it in place, so
    // this keeps every pass computing from identical initial state.
    auto run_nw_pass = [&]() {
        bench_lap_reset(&acc_k1);
        bench_lap_reset(&acc_k2);

        HIP_CHECK(hipMemcpy(d_matrix, h_matrix,
                            matrix_elems * sizeof(int),
                            hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        // Phase 1: upper-left triangle (nw_kernel1)
        for (int diag = 0; diag < num_blocks; diag++) {
            int n_blocks = diag + 1;
            bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
                nw_kernel1<<<dim3(n_blocks), dim3(block_size), shared_mem, 0>>>(
                    d_seq, d_matrix, cols, penalty, diag, block_size);
            }));
        }

        // Phase 2: lower-right triangle (nw_kernel2)
        for (int diag = 0; diag < num_blocks - 1; diag++) {
            int n_blocks = num_blocks - 1 - diag;
            bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
                nw_kernel2<<<dim3(n_blocks), dim3(block_size), shared_mem, 0>>>(
                    d_seq, d_matrix, cols, penalty, diag, num_blocks, block_size);
            }));
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_nw_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_nw_pass();
        bench_emit_iter("nw_kernel1", i, &acc_k1, params);
        bench_emit_iter("nw_kernel2", i, &acc_k2, params);
        total_gpu_ms += acc_k1.gpu_ms + acc_k2.gpu_ms;
        fprintf(stderr, "pass %d: nw_kernel1 gpu = %.3f ms, nw_kernel2 gpu = %.3f ms\n",
                i, acc_k1.gpu_ms, acc_k2.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    HIP_CHECK(hipFree(d_seq));
    HIP_CHECK(hipFree(d_matrix));
    free(h_seq);
    free(h_matrix);

    return 0;
}
