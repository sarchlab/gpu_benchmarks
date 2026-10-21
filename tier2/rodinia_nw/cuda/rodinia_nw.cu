// Rodinia 3.1 Needleman-Wunsch: two seed-7 sequences, BLOSUM62 scoring,
// and a complete tiled DP wavefront. Independent CUDA implementation.
// Input setup, result download and full integer CPU verification are untimed.
// One entry warmer precedes the first timed launch; both dependent phases are
// accumulated per kernel and their GPU event times are summed in the summary.
// --sequence_length N, --block_size B, --penalty P; BENCH_PARAM_* overrides argv.
// Defaults: N=2048, B=16, P=2. Manual N rounds up to a multiple of B.
// See README for the pinned source, effective work, verification and limits.

#include "../../../common_cuda/bench_timing.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

/* BLOSUM62 table from Rodinia 3.1, pin366b283456506ef2fe2c2b7dc5e83e941ffc5524.
LICENSE TERMS

Copyright (c)2008-2011 University of Virginia
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted without royalty fees or other restrictions, provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Virginia, the Dept. of Computer Science, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF VIRGINIA OR THE SOFTWARE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

static const int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

struct NwGeometry {
    int blocks;
    int length;
    int columns;
    size_t cells;
    size_t shared_bytes;
};

// Keep every linear kernel index and every candidate score in signed-int range.
static bool nw_geometry(int requested, int block, int penalty, NwGeometry* out) {
    if (requested <= 0 || block <= 0 || penalty <= 0) return false;
    const long long blocks = (static_cast<long long>(requested) + block - 1) / block;
    const long long length = blocks * block;
    const long long columns = length + 1;
    if (columns > INT_MAX / columns) return false;
    // A diagonal path plus edge gaps bounds each DP value below by
    // -length*max(penalty,4), and above by 11*length. Include one further
    // gap/substitution so even losing recurrence candidates cannot overflow.
    if (columns * std::max(penalty, 11) > INT_MAX) return false;
    out->blocks = static_cast<int>(blocks);
    out->length = static_cast<int>(length);
    out->columns = static_cast<int>(columns);
    out->cells = static_cast<size_t>(columns * columns);
    out->shared_bytes = (static_cast<size_t>(block + 1) * (block + 1)
                         + static_cast<size_t>(block) * block) * sizeof(int);
    return true;
}

static void nw_initialize(int length, int penalty, int* row_sequence,
                          int* column_sequence, int* reference, int* matrix) {
    const int columns = length + 1;
    std::fill(matrix, matrix + static_cast<size_t>(columns) * columns, 0);
    std::fill(reference, reference + static_cast<size_t>(columns) * columns, 0);
    row_sequence[0] = column_sequence[0] = 0;
    // Preserve the pinned driver's order: all row draws, then all column draws.
    srand(7);
    for (int i = 1; i <= length; ++i) row_sequence[i] = rand() % 10 + 1;
    for (int j = 1; j <= length; ++j) column_sequence[j] = rand() % 10 + 1;
    for (int i = 1; i <= length; ++i) {
        matrix[i * columns] = -i * penalty;
        matrix[i] = -i * penalty;
        for (int j = 1; j <= length; ++j)
            reference[i * columns + j] = blosum62[row_sequence[i]][column_sequence[j]];
    }
}

// Independent row-major CPU recurrence: use the retained sequences directly,
// not the uploaded reference matrix or any measured DP value as recurrence input.
static size_t nw_verify(const int* measured, const int* row_sequence,
                        const int* column_sequence, int length, int penalty) {
    const int columns = length + 1;
    std::vector<int> previous(columns), current(columns);
    size_t errors = 0;
    auto compare = [&](int i, int j, int expected) {
        const int actual = measured[i * columns + j];
        if (actual != expected) {
            if (errors < 10)
                fprintf(stderr, "Mismatch at (%d,%d): GPU=%d CPU=%d\n", i, j, actual, expected);
            ++errors;
        }
    };
    for (int j = 0; j <= length; ++j) {
        previous[j] = -j * penalty;
        compare(0, j, previous[j]);
    }
    for (int i = 1; i <= length; ++i) {
        current[0] = -i * penalty;
        compare(i, 0, current[0]);
        for (int j = 1; j <= length; ++j) {
            const int diagonal = previous[j - 1] + blosum62[row_sequence[i]][column_sequence[j]];
            current[j] = std::max(diagonal, std::max(previous[j] - penalty, current[j - 1] - penalty));
            compare(i, j, current[j]);
        }
        previous.swap(current);
    }
    return errors;
}

// ---------------------------------------------------------------------------
// Kernel 1: upper-left triangle of the block dependency graph
//
// For diagonal block_idx = 0..num_blocks-1, dispatches (block_idx+1) blocks.
// Block bx handles block position (row = block_idx - bx, col = bx).
// Each block uses shared memory to fill its (block_size x block_size) tile.
// ---------------------------------------------------------------------------
__global__ void nw_kernel1(const int* __restrict__ d_reference,
                            int* __restrict__ d_matrix,
                            int cols,
                            int penalty,
                            int block_idx,
                            int block_size)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Filter out blocks beyond diagonal
    if (bx > block_idx) return;

    int b_row = block_idx - bx;
    int b_col = bx;

    // Global starting position (1-indexed within the matrix)
    int begin_row = b_row * block_size + 1;
    int begin_col = b_col * block_size + 1;

    // Shared memory layout:
    //   temp[0..(block_size+1)^2-1] : local DP tile (block_size+1 x block_size+1)
    //   ref_shared[0..block_size^2-1] : BLOSUM62 reference tile
    extern __shared__ int shared[];
    int* temp       = shared;
    int* ref_shared = &shared[(block_size + 1) * (block_size + 1)];

    // Cooperatively load the precomputed BLOSUM62 tile, as in Rodinia.
    for (int row = 0; row < block_size; ++row)
        ref_shared[row * block_size + tx] =
            d_reference[(begin_row + row) * cols + begin_col + tx];
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

            int match = temp[idx_upleft] +
                        ref_shared[(i - 1) * block_size + j - 1];
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
__global__ void nw_kernel2(const int* __restrict__ d_reference,
                            int* __restrict__ d_matrix,
                            int cols,
                            int penalty,
                            int block_idx,
                            int num_blocks,
                            int block_size)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int blocks_on_diag = num_blocks - 1 - block_idx;
    if (bx >= blocks_on_diag) return;

    int b_row = num_blocks - 1 - bx;
    int b_col = block_idx + 1 + bx;

    int begin_row = b_row * block_size + 1;
    int begin_col = b_col * block_size + 1;

    extern __shared__ int shared[];
    int* temp       = shared;
    int* ref_shared = &shared[(block_size + 1) * (block_size + 1)];

    for (int row = 0; row < block_size; ++row)
        ref_shared[row * block_size + tx] =
            d_reference[(begin_row + row) * cols + begin_col + tx];
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
                        ref_shared[(i - 1) * block_size + j - 1];
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

int main(int argc, char** argv) try {
    setvbuf(stdout, NULL, _IONBF, 0);
    int seq_len = parseIntParam(argc, argv, "--sequence_length", 2048);
    int block_size = parseIntParam(argc, argv, "--block_size", 16);
    int penalty = parseIntParam(argc, argv, "--penalty", 2);
    if (const char* value = getenv("BENCH_PARAM_sequence_length")) seq_len = atoi(value);
    if (const char* value = getenv("BENCH_PARAM_block_size")) block_size = atoi(value);
    if (const char* value = getenv("BENCH_PARAM_penalty")) penalty = atoi(value);

    NwGeometry geometry;
    if (!nw_geometry(seq_len, block_size, penalty, &geometry)) {
        fprintf(stderr, "Invalid NW dimensions/penalty or signed index/score range\n");
        return EXIT_FAILURE;
    }
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    if (block_size > prop.maxThreadsPerBlock || block_size > prop.maxThreadsDim[0] ||
        geometry.blocks > prop.maxGridSize[0] || geometry.shared_bytes > prop.sharedMemPerBlock) {
        fprintf(stderr, "NW block/grid/shared-memory geometry exceeds device limits\n");
        return EXIT_FAILURE;
    }
    const int num_blocks = geometry.blocks;
    const int padded_len = geometry.length;
    const int cols = geometry.columns;
    const size_t matrix_bytes = geometry.cells * sizeof(int);
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Sequence length: %d (effective %d) | Block size: %d | Penalty: %d\n",
            seq_len, padded_len, block_size, penalty);

    std::vector<int> row_sequence(cols), column_sequence(cols);
    std::vector<int> h_reference(geometry.cells), h_matrix(geometry.cells);
    nw_initialize(padded_len, penalty, row_sequence.data(), column_sequence.data(),
                  h_reference.data(), h_matrix.data());
    int *d_reference, *d_matrix;
    CUDA_CHECK(cudaMalloc(&d_reference, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_reference, h_reference.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));

    char params[512];
    snprintf(params, sizeof(params),
             "\"sequence_length\":%d,\"block_size\":%d,\"penalty\":%d,"
             "\"effective_sequence_length\":%d,\"matrix_cells\":%zu,\"total_cell_updates\":%lld,"
             "\"input_seed\":7,\"algorithm\":\"rodinia31_blosum62_v1\","
             "\"verification\":\"full_dp_cpu_v1\"",
             seq_len, block_size, penalty, padded_len, geometry.cells,
             static_cast<long long>(padded_len) * padded_len);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_k1, acc_k2;
    bench_lap_reset(&acc_k1);
    bench_lap_reset(&acc_k2);
    // One complete fill; preserve dependencies and cache reuse between phases.
    bench_pass_warm();
    for (int diag = 0; diag < num_blocks; ++diag) {
        bench_lap_add(&acc_k1, bench_time_op(&timer, [&]() {
            nw_kernel1<<<dim3(diag + 1), dim3(block_size), geometry.shared_bytes, 0>>>(
                d_reference, d_matrix, cols, penalty, diag, block_size);
        }));
    }
    for (int diag = 0; diag < num_blocks - 1; ++diag) {
        bench_lap_add(&acc_k2, bench_time_op(&timer, [&]() {
            nw_kernel2<<<dim3(num_blocks - 1 - diag), dim3(block_size), geometry.shared_bytes, 0>>>(
                d_reference, d_matrix, cols, penalty, diag, num_blocks, block_size);
        }));
    }
    char kernel_params[640];
    const long long upper_tiles = static_cast<long long>(num_blocks) * (num_blocks + 1) / 2;
    const long long lower_tiles = static_cast<long long>(num_blocks) * (num_blocks - 1) / 2;
    snprintf(kernel_params, sizeof(kernel_params), "%s,\"kernel_launches\":%d,\"cell_updates\":%lld",
             params, num_blocks, upper_tiles * block_size * block_size);
    bench_emit_iter("nw_kernel1", 0, &acc_k1, kernel_params);
    if (num_blocks > 1) {
        snprintf(kernel_params, sizeof(kernel_params), "%s,\"kernel_launches\":%d,\"cell_updates\":%lld",
                 params, num_blocks - 1, lower_tiles * block_size * block_size);
        bench_emit_iter("nw_kernel2", 0, &acc_k2, kernel_params);
    }
    const double total_gpu_ms = acc_k1.gpu_ms + acc_k2.gpu_ms;
    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Download the actual measured matrix; no second GPU verification run.
    CUDA_CHECK(cudaMemcpy(h_matrix.data(), d_matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_reference));
    CUDA_CHECK(cudaFree(d_matrix));
    const size_t errors = nw_verify(h_matrix.data(), row_sequence.data(),
                                   column_sequence.data(), padded_len, penalty);
    fprintf(stderr, "%s (verified %zu DP cells including borders, %zu mismatches)\n",
            errors == 0 ? "PASS" : "FAIL", geometry.cells, errors);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& error) {
    fprintf(stderr, "FAIL: NW host/reference storage unavailable: %s\n", error.what());
    return EXIT_FAILURE;
}
