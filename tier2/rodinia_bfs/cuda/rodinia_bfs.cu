// rodinia_bfs.cu — Rodinia BFS benchmark (CUDA, self-contained)
//
// Breadth-First Search on a randomly-generated CSR (Compressed Sparse Row)
// graph. Two kernels per BFS level:
//   1. bfs_kernel  — for each frontier node, write cost to unvisited neighbors
//   2. bfs_update  — promote newly-discovered nodes into the next frontier
//
// Native CUDA port of ../rodinia_bfs.hip (same kernels, driver and
// output). Compiles with nvcc; no architecture-specific features are used
// (any sm_70+ GPU; intended targets are A100 sm_80 and H100 sm_90).
//
// Usage:
//   ./rodinia_bfs [--num_nodes N] [--edges_per_node E] [--block_size B]
//
//   --num_nodes N       Number of graph nodes            (default: 65536)
//   --edges_per_node E  Out-degree of each node          (default: 20)
//   --block_size B      GPU thread-block size            (default: 256)
//
// Environment variables (override argv):
//   BENCH_PARAM_num_nodes, BENCH_PARAM_edges_per_node, BENCH_PARAM_block_size
//
// Output (stdout): one {"type":"kernel",...} JSON line per
// kernel (bfs_kernel, bfs_update) carrying that kernel's accumulated GPU
// event time and host time within the pass, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Bandwidth /
// edges-per-second are derived downstream from time_ms and params.

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "cpu_reference.h"

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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
// BFS kernels
// ---------------------------------------------------------------------------

// Pass 1: the visited mask is immutable until bfs_update. Frontier vertices
// are already visited: their costs are only read. Only unvisited costs are
// written, and all competing stores are atomic and carry the same level+1.
// This separates ordinary reads from atomic writes by vertex ownership.
__global__ void bfs_kernel(const int* __restrict__ row_offsets,
                           const int* __restrict__ col_indices,
                           int*       cost,
                           const int* frontier,
                           const int* visited,
                           int        num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes && frontier[tid]) {
        int row_start = row_offsets[tid];
        int row_end   = row_offsets[tid + 1];
        int new_cost  = cost[tid] + 1;
        for (int e = row_start; e < row_end; ++e) {
            int nb = col_indices[e];
            if (!visited[nb]) {
                // Eligibility is immutable throughout this BFS level. Every
                // competing parent makes one atomic attempt; none is skipped
                // because another thread has already stored the same distance.
                atomicExch(&cost[nb], new_cost);
            }
        }
    }
}

// Pass 2: promote newly-discovered nodes into next frontier; signal if any
__global__ void bfs_update(int*       frontier,
                           const int* cost,
                           int*       visited,
                           int*       updated,
                           int        num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes) {
        int was_visited = visited[tid];
        frontier[tid] = 0;
        if (!was_visited && cost[tid] >= 0) {
            visited[tid]  = 1;
            frontier[tid] = 1;
            atomicExch(updated, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Graph generation (CSR, random)
// ---------------------------------------------------------------------------

static void generate_random_graph(int num_nodes, int edges_per_node,
                                  int** h_row_offsets, int** h_col_indices,
                                  int*  num_edges_out) {
    int num_edges = num_nodes * edges_per_node;
    *h_row_offsets = (int*)malloc(((size_t)num_nodes + 1) * sizeof(int));
    *h_col_indices = (int*)malloc((size_t)num_edges * sizeof(int));

    srand(42);
    int offset = 0;
    for (int i = 0; i < num_nodes; ++i) {
        (*h_row_offsets)[i] = offset;
        for (int j = 0; j < edges_per_node; ++j) {
            (*h_col_indices)[offset + j] = rand() % num_nodes;
        }
        offset += edges_per_node;
    }
    (*h_row_offsets)[num_nodes] = offset;
    *num_edges_out = num_edges;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int num_nodes      = parseIntParam(argc, argv, "--num_nodes",      65536);
    int edges_per_node = parseIntParam(argc, argv, "--edges_per_node", 20);
    int block_size     = parseIntParam(argc, argv, "--block_size",     256);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_num_nodes");
    if (env_val) num_nodes = atoi(env_val);
    env_val = getenv("BENCH_PARAM_edges_per_node");
    if (env_val) edges_per_node = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    if (num_nodes <= 0 || edges_per_node < 0 || block_size <= 0) {
        fprintf(stderr, "Invalid BFS parameters: num_nodes and block_size must be positive; "
                        "edges_per_node must be nonnegative\n");
        return EXIT_FAILURE;
    }
    const unsigned long long edge_count =
        (unsigned long long)num_nodes * (unsigned long long)edges_per_node;
    if (edge_count > (unsigned long long)INT_MAX) {
        fprintf(stderr, "BFS edge count exceeds INT_MAX\n");
        return EXIT_FAILURE;
    }
    const unsigned long long grid_blocks =
        ((unsigned long long)num_nodes + (unsigned long long)block_size - 1) /
        (unsigned long long)block_size;
    // Inactive padded threads also compute tid before the bounds check.
    if (grid_blocks * (unsigned long long)block_size - 1 > (unsigned long long)INT_MAX) {
        fprintf(stderr, "BFS padded vertex indices exceed INT_MAX\n");
        return EXIT_FAILURE;
    }

    // Generate random graph
    int* h_row_offsets = nullptr;
    int* h_col_indices = nullptr;
    int  num_edges     = 0;
    generate_random_graph(num_nodes, edges_per_node,
                          &h_row_offsets, &h_col_indices, &num_edges);

    // Host arrays for BFS state
    int* h_cost     = (int*)malloc((size_t)num_nodes * sizeof(int));
    int* h_frontier = (int*)malloc((size_t)num_nodes * sizeof(int));
    int* h_visited  = (int*)malloc((size_t)num_nodes * sizeof(int));

    // Device allocations
    int *d_row_offsets, *d_col_indices;
    int *d_cost, *d_frontier, *d_visited, *d_updated;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, ((size_t)num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, (size_t)num_edges        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cost,        (size_t)num_nodes        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,    (size_t)num_nodes        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited,     (size_t)num_nodes        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_updated,     sizeof(int)));

    // Copy graph structure to device (constant across iterations)
    CUDA_CHECK(cudaMemcpy(d_row_offsets, h_row_offsets,
                        ((size_t)num_nodes + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, h_col_indices,
                        (size_t)num_edges * sizeof(int),
                        cudaMemcpyHostToDevice));

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Nodes: %d  |  Edges: %d  |  Block: %d\n\n",
            num_nodes, num_edges, block_size);

    dim3 block(block_size);
    dim3 grid((unsigned int)grid_blocks);

    char params[384];
    unsigned long long bfs_levels = 0;

    // Per-kernel accumulators: each BFS pass launches both kernels once per
    // BFS level; the accumulated GPU/host time of all levels within one pass
    // becomes that pass's record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_kernel, acc_update;

    // One full BFS pass. State is re-initialized at the start of every pass
    // (untimed), so each pass measures the same traversal regardless of what
    // earlier passes left behind.
    auto run_bfs_pass = [&]() {
        bfs_levels = 0;
        bench_lap_reset(&acc_kernel);
        bench_lap_reset(&acc_update);

        for (int i = 0; i < num_nodes; ++i) {
            h_cost[i]     = -1;
            h_frontier[i] = 0;
            h_visited[i]  = 0;
        }
        h_cost[0]     = 0;
        h_frontier[0] = 1;
        h_visited[0]  = 1;

        CUDA_CHECK(cudaMemcpy(d_cost,     h_cost,     (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_visited,  h_visited,  (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        int h_updated = 1;
        while (h_updated) {
            h_updated = 0;
            CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice));

            bench_lap_add(&acc_kernel, bench_time_op(&timer, [&]() {
                bfs_kernel<<<grid, block>>>(d_row_offsets, d_col_indices, d_cost, d_frontier, d_visited, num_nodes);
            }));

            h_updated = 0;
            CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice));

            bench_lap_add(&acc_update, bench_time_op(&timer, [&]() {
                bfs_update<<<grid, block>>>(d_frontier, d_cost, d_visited, d_updated, num_nodes);
            }));

            CUDA_CHECK(cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost));
            ++bfs_levels;
        }
    };

    run_bfs_pass();

    // Verify every distance produced by the measured traversal, after timing.
    CUDA_CHECK(cudaMemcpy(h_cost, d_cost, (size_t)num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    BfsWork work;
    const std::vector<int> reference = bfs_reference(h_row_offsets, h_col_indices, num_nodes, &work);
    int errors = 0;
    for (int v = 0; v < num_nodes; ++v) {
        if (h_cost[v] != reference[v]) {
            if (errors < 10)
                fprintf(stderr, "MISMATCH cost[%d]: gpu=%d cpu=%d\n", v, h_cost[v], reference[v]);
            ++errors;
        }
    }
    if (bfs_levels != work.levels) {
        if (errors < 10)
            fprintf(stderr, "MISMATCH BFS levels: gpu=%llu cpu=%llu\n", bfs_levels, work.levels);
        ++errors;
    }
    fprintf(stderr, "BFS verification: %s (%d distances, %d mismatches)\n",
            errors ? "FAIL" : "PASS", num_nodes, errors);

    snprintf(params, sizeof(params),
             "\"num_nodes\":%d,\"edges_per_node\":%d,\"block_size\":%d,"
             "\"verification\":\"full_cpu_v1\",\"discovery_policy\":\"level_visited_atomic_v1\","
             "\"bfs_levels\":%llu,\"frontier_edges\":%llu,\"discovery_atomics\":%llu,\"promotion_atomics\":%llu",
             num_nodes, edges_per_node, block_size, bfs_levels,
             work.frontier_edges, work.discovery_atomics, work.promotion_atomics);
    fprintf(stderr, "BFS work: levels=%llu frontier_edges=%llu discovery_atomics=%llu promotion_atomics=%llu\n",
            bfs_levels, work.frontier_edges, work.discovery_atomics, work.promotion_atomics);

    double total_gpu_ms = 0.0;
    bench_emit_iter("bfs_kernel", 0, &acc_kernel, params);
    bench_emit_iter("bfs_update", 0, &acc_update, params);
    total_gpu_ms += acc_kernel.gpu_ms + acc_update.gpu_ms;
    fprintf(stderr, "bfs_kernel gpu = %.3f ms, bfs_update gpu = %.3f ms\n",
            acc_kernel.gpu_ms, acc_update.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over 1 pass\n",
            total_gpu_ms);

    // Cleanup
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_cost));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_updated));
    free(h_row_offsets);
    free(h_col_indices);
    free(h_cost);
    free(h_frontier);
    free(h_visited);

    return errors ? 1 : 0;
}
