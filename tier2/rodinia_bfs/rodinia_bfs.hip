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
//   ./rodinia_bfs [--num_nodes N] [--edges_per_node E] [--block_size B] [--iterations I]
//
//   --num_nodes N       Number of graph nodes            (default: 65536)
//   --edges_per_node E  Out-degree of each node          (default: 20)
//   --block_size B      GPU thread-block size            (default: 256)
//   --iterations I      Timed benchmark iterations       (default: 20)
//
// Output (stdout): CSV row —
//   kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
// Output (stderr): Effective bandwidth in GB/s

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Inlined bench_common_hip.h utilities
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

static int parseIterations(int argc, char** argv) {
    int iters = 20;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--iterations") == 0) {
            int v = atoi(argv[i + 1]);
            if (v >= 1) iters = v;
            break;
        }
    }
    return iters;
}

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


struct BenchmarkTimer {
    hipEvent_t start, stop;
    BenchmarkTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~BenchmarkTimer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    void record_start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start, stream));
    }
    void record_stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

template <typename Func>
static BenchResult runBenchmark(const char* kernel_name,
                                const char* problem_size,
                                int         iterations,
                                Func        func) {
    BenchmarkTimer timer;

    // Warmup
    func();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        timer.record_start();
        func();
        timer.record_stop();
        times[i] = static_cast<double>(timer.elapsed_ms());
    }

    double sum = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum / iterations;

    double variance = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iterations > 1) ? sqrt(variance / (iterations - 1)) : 0.0;

    BenchResult r;
    r.kernel_name  = kernel_name;
    r.problem_size = problem_size;
    r.iterations   = iterations;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;
    return r;
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
    int iters          = parseIterations(argc, argv);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_num_nodes");
    if (env_val) num_nodes = atoi(env_val);
    env_val = getenv("BENCH_PARAM_edges_per_node");
    if (env_val) edges_per_node = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iterations");
    if (env_val) iters = atoi(env_val);

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
    fprintf(stderr, "Nodes: %d  |  Edges: %d  |  Block: %d  |  Iterations: %d\n\n",
            num_nodes, num_edges, block_size, iters);

    dim3 block(block_size);
    dim3 grid((num_nodes + block_size - 1) / block_size);

    char problemSize[64];
    snprintf(problemSize, sizeof(problemSize), "%d", num_nodes);

    // Bandwidth accounting:
    //   Each BFS iteration reads row_offsets (num_nodes+1 ints),
    //   col_indices (num_edges ints), cost (num_nodes ints),
    //   frontier (num_nodes ints), visited (num_nodes ints)
    //   and writes cost (num_edges worst-case), frontier, visited, updated.
    // We estimate conservatively as 2 * (row_offsets + col_indices + 3*nodes)
    long long bytes_per_bfs =
        2LL * ((long long)(num_nodes + 1 + num_edges + 3 * num_nodes) * sizeof(int));

    BenchResult r = runBenchmark("rodinia_bfs", problemSize, iters, [&]() {
        // Re-initialize BFS state for each timed run
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

            hipLaunchKernelGGL(bfs_kernel, grid, block, 0, 0,
                               d_row_offsets, d_col_indices,
                               d_cost, d_frontier, num_nodes);
            HIP_CHECK(hipDeviceSynchronize());

            h_updated = 0;
            HIP_CHECK(hipMemcpy(d_updated, &h_updated, sizeof(int), hipMemcpyHostToDevice));

            hipLaunchKernelGGL(bfs_update, grid, block, 0, 0,
                               d_frontier, d_cost, d_visited, d_updated, num_nodes);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(&h_updated, d_updated, sizeof(int), hipMemcpyDeviceToHost));
        }
    });

    double bw_gb = (double)bytes_per_bfs / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Effective bandwidth: %.2f GB/s  (avg %.4f ms/BFS)\n",
            bw_gb, r.avg_ms);

    // JSON-lines output
    double edges_per_sec = (double)num_edges / (r.avg_ms * 1e-3);
    printf("{\"type\":\"kernel\",\"name\":\"bfs_kernel\",\"time_ms\":%.6f,"
           "\"params\":{\"num_nodes\":%d,\"edges_per_node\":%d,\"block_size\":%d,"
           "\"iterations\":%d}}\n",
           r.avg_ms, num_nodes, edges_per_node, block_size, iters);
    printf("{\"type\":\"kernel\",\"name\":\"bfs_update\",\"time_ms\":%.6f,"
           "\"params\":{\"num_nodes\":%d,\"edges_per_node\":%d,\"block_size\":%d,"
           "\"iterations\":%d}}\n",
           r.avg_ms, num_nodes, edges_per_node, block_size, iters);
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"bandwidth_gbps\",\"value\":%.2f},"
           "{\"name\":\"edges_per_sec\",\"value\":%.2f}]}\n",
           r.avg_ms, bw_gb, edges_per_sec);

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
