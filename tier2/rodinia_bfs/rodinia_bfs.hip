// rodinia_bfs.hip — Rodinia BFS benchmark (HIP, self-contained)
//
// Breadth-First Search on a randomly-generated CSR (Compressed Sparse Row)
// graph. Two kernels per BFS level:
//   1. bfs_kernel  — for each frontier node, write cost to unvisited neighbors
//   2. bfs_update  — promote newly-discovered nodes into the next frontier
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
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
//   BENCH_PARAM_num_warmup   (warmup BFS passes, default 3)
//   BENCH_PARAM_num_iters    (timed BFS passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (bfs_kernel, bfs_update) carrying that kernel's accumulated GPU
// event time and host time within the pass, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Bandwidth /
// edges-per-second are derived downstream from time_ms and params.

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
// BFS kernels
// ---------------------------------------------------------------------------

// Pass 1: for each frontier node, write cost+1 to unvisited neighbors
__global__ void bfs_kernel(const int* __restrict__ row_offsets,
                           const int* __restrict__ col_indices,
                           int*       cost,
                           const int* frontier,
                           int        num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes && frontier[tid]) {
        int row_start = row_offsets[tid];
        int row_end   = row_offsets[tid + 1];
        int new_cost  = cost[tid] + 1;
        for (int e = row_start; e < row_end; ++e) {
            int nb = col_indices[e];
            if (cost[nb] < 0) {
                cost[nb] = new_cost;  // may race but harmless (same value)
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
            *updated = 1;
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
    *h_row_offsets = (int*)malloc((size_t)(num_nodes + 1) * sizeof(int));
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
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

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

    HIP_CHECK(hipMalloc(&d_row_offsets, (size_t)(num_nodes + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, (size_t)num_edges        * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cost,        (size_t)num_nodes        * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_frontier,    (size_t)num_nodes        * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_visited,     (size_t)num_nodes        * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_updated,     sizeof(int)));

    // Copy graph structure to device (constant across iterations)
    HIP_CHECK(hipMemcpy(d_row_offsets, h_row_offsets,
                        (size_t)(num_nodes + 1) * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, h_col_indices,
                        (size_t)num_edges * sizeof(int),
                        hipMemcpyHostToDevice));

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Nodes: %d  |  Edges: %d  |  Block: %d\n\n",
            num_nodes, num_edges, block_size);

    dim3 block(block_size);
    dim3 grid((num_nodes + block_size - 1) / block_size);

    char params[160];
    snprintf(params, sizeof(params),
             "\"num_nodes\":%d,\"edges_per_node\":%d,\"block_size\":%d",
             num_nodes, edges_per_node, block_size);

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

        HIP_CHECK(hipMemcpy(d_cost,     h_cost,     (size_t)num_nodes * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_frontier, h_frontier, (size_t)num_nodes * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_visited,  h_visited,  (size_t)num_nodes * sizeof(int), hipMemcpyHostToDevice));

        int h_updated = 1;
        while (h_updated) {
            h_updated = 0;
            HIP_CHECK(hipMemcpy(d_updated, &h_updated, sizeof(int), hipMemcpyHostToDevice));

            bench_lap_add(&acc_kernel, bench_time_op(&timer, [&]() {
                bfs_kernel<<<grid, block>>>(d_row_offsets, d_col_indices, d_cost, d_frontier, num_nodes);
            }));

            h_updated = 0;
            HIP_CHECK(hipMemcpy(d_updated, &h_updated, sizeof(int), hipMemcpyHostToDevice));

            bench_lap_add(&acc_update, bench_time_op(&timer, [&]() {
                bfs_update<<<grid, block>>>(d_frontier, d_cost, d_visited, d_updated, num_nodes);
            }));

            HIP_CHECK(hipMemcpy(&h_updated, d_updated, sizeof(int), hipMemcpyDeviceToHost));
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_bfs_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_bfs_pass();
        bench_emit_iter("bfs_kernel", i, &acc_kernel, params);
        bench_emit_iter("bfs_update", i, &acc_update, params);
        total_gpu_ms += acc_kernel.gpu_ms + acc_update.gpu_ms;
        fprintf(stderr, "pass %d: bfs_kernel gpu = %.3f ms, bfs_update gpu = %.3f ms\n",
                i, acc_kernel.gpu_ms, acc_update.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_row_offsets));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_cost));
    HIP_CHECK(hipFree(d_frontier));
    HIP_CHECK(hipFree(d_visited));
    HIP_CHECK(hipFree(d_updated));
    free(h_row_offsets);
    free(h_col_indices);
    free(h_cost);
    free(h_frontier);
    free(h_visited);

    return 0;
}
