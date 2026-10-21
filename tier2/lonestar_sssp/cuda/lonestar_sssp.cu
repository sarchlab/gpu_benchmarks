// lonestar_sssp.cu — Single-Source Shortest Path (Bellman-Ford) benchmark (CUDA)
//
// Iterative edge relaxation on a random sparse graph stored in CSR format.
// Each round reads an immutable distance snapshot and atomically relaxes
// reachable outgoing edges into the result; repeats until no changes.
//
// Native CUDA port of ../hip/lonestar_sssp.hip; compiles with nvcc.
//
// Usage:
//   ./lonestar_sssp [--vertices N] [--degree D] [--block_size B]
//
//   --vertices N     Number of vertices (default: 65536)
//   --degree D       Average degree per vertex (default: 16)
//   --block_size B   Threads per block (default: 256)
//
// Environment variables (override argv):
//   BENCH_PARAM_vertices, BENCH_PARAM_degree, BENCH_PARAM_block_size
//
// Output: one accumulated timing record per kernel (snapshot and relaxation),
// then their summed GPU time. Parameters identify the input, schedule, complete
// verification, actual rounds, and derived logical edge-relaxation attempts.
// Output (stderr): human-readable results

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include "cpu_reference.h"

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (xorshift32)
// ---------------------------------------------------------------------------

static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// ---------------------------------------------------------------------------
// Generate random CSR graph
// ---------------------------------------------------------------------------

static void generate_csr_graph(int N, int avg_degree, uint32_t seed,
                                std::vector<int>& row_offsets,
                                std::vector<int>& col_indices,
                                std::vector<int>& weights) {
    uint32_t rng = seed;
    row_offsets.resize(N + 1);
    // First pass: determine edges per vertex
    std::vector<int> degrees(N);
    for (int i = 0; i < N; ++i) {
        // Poisson-like: use avg_degree +/- some variance
        int d = avg_degree / 2 + (int)(xorshift32(rng) % (avg_degree + 1));
        if (d < 1) d = 1;
        degrees[i] = d;
    }

    // Build row_offsets
    row_offsets[0] = 0;
    for (int i = 0; i < N; ++i)
        row_offsets[i + 1] = row_offsets[i] + degrees[i];

    int total_edges = row_offsets[N];
    col_indices.resize(total_edges);
    weights.resize(total_edges);

    // Fill edges
    for (int i = 0; i < N; ++i) {
        for (int e = row_offsets[i]; e < row_offsets[i + 1]; ++e) {
            col_indices[e] = (int)(xorshift32(rng) % (uint32_t)N);
            weights[e] = 1 + (int)(xorshift32(rng) % 100); // weight 1-100
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Bellman-Ford edge relaxation
// Each thread processes one vertex's outgoing edges
// ---------------------------------------------------------------------------

__global__ void sssp_snapshot_kernel(const int* dist, int* previous, int N) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N) previous[u] = dist[u];
}

__global__ void bellman_ford_kernel(const int* __restrict__ row_offsets,
                                    const int* __restrict__ col_indices,
                                    const int* __restrict__ weights,
                                    const int* __restrict__ previous,
                                    int* __restrict__ dist,
                                    int* __restrict__ changed,
                                    int N) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    int d_u = previous[u];
    if (d_u == INT_MAX) return; // unreachable, skip

    int start = row_offsets[u];
    int end   = row_offsets[u + 1];

    bool updated = false;
    for (int e = start; e < end; ++e) {
        int v = col_indices[e];
        int w = weights[e];
        int new_dist = d_u + w;
        // Every reachable edge makes one attempt. Only immutable previous
        // values decide whether this vertex signals a change, so flag work
        // does not depend on which competing atomicMin wins first.
        atomicMin(&dist[v], new_dist);
        if (new_dist < previous[v]) updated = true;
    }
    if (updated) atomicExch(changed, 1);
}

// ---------------------------------------------------------------------------
// GPU SSSP: run Bellman-Ford iterations
// ---------------------------------------------------------------------------

static int run_sssp_gpu(int N,
                         int* d_row_offsets, int* d_col_indices,
                         int* d_weights, int* d_dist, int* d_previous, int* d_changed,
                         int source, int blkSize,
                         bench_timer_t* timer, bench_lap_t* snapshot_acc, bench_lap_t* acc) {
    bench_lap_reset(snapshot_acc);
    bench_lap_reset(acc);

    // Reset per-pass state (untimed): distances back to INT_MAX / source = 0
    // so every pass performs the same convergence work.
    std::vector<int> h_dist(N, INT_MAX);
    h_dist[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = blkSize;
    int gridSize  = (int)(((long long)N + blockSize - 1) / blockSize);
    int max_iters = N; // upper bound
    int total_iters = 0;

    /* Cold-cache calibration: evict the LLC (and hold the clock high) after
     * the per-pass H2D re-init so the pass's first kernel launch runs cold.
     * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
    bench_pass_warm();

    for (int iter = 0; iter < max_iters; ++iter) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        // This copy is required by the synchronous algorithm, hence timed.
        bench_lap_add(snapshot_acc, bench_time_op(timer, [&]() {
            sssp_snapshot_kernel<<<dim3(gridSize), dim3(blockSize)>>>(d_dist, d_previous, N);
        }));
        bench_lap_add(acc, bench_time_op(timer, [&]() {
            bellman_ford_kernel<<<dim3(gridSize), dim3(blockSize)>>>(
                d_row_offsets, d_col_indices, d_weights, d_previous, d_dist, d_changed, N);
        }));

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        total_iters++;
        if (!h_changed) break;
    }
    return total_iters;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int vertices   = 65536;
    int degree     = 16;
    int block_size = 256;

    // Command-line fallback
    vertices   = parseIntParam(argc, argv, "--vertices", vertices);
    degree     = parseIntParam(argc, argv, "--degree", degree);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_vertices");
    if (env_val) vertices = atoi(env_val);
    env_val = getenv("BENCH_PARAM_degree");
    if (env_val) degree = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    int N       = vertices;
    int avg_deg = degree;
    int source  = 0;

    if (N <= 0 || avg_deg < 0 || block_size <= 0) {
        fprintf(stderr, "SSSP vertices and block_size must be positive; degree must be nonnegative\n");
        return EXIT_FAILURE;
    }
    // Bound every generated CSR prefix, including the degree expression itself.
    long long max_degree = (long long)avg_deg / 2 + avg_deg;
    if (max_degree < 1) max_degree = 1;
    if ((long long)N * max_degree > INT_MAX) {
        fprintf(stderr, "SSSP maximum generated edge count exceeds INT_MAX\n");
        return EXIT_FAILURE;
    }
    // At most N rounds, with weights 1..100. Keep every candidate below the
    // unreachable sentinel, including the final no-change round's attempts.
    if ((long long)N * 100 >= INT_MAX) {
        fprintf(stderr, "SSSP maximum distance candidate reaches INT_MAX\n");
        return EXIT_FAILURE;
    }

    // Generate CSR graph
    std::vector<int> row_offsets, col_indices, weights;
    generate_csr_graph(N, avg_deg, 42, row_offsets, col_indices, weights);
    int num_edges = row_offsets[N];

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "SSSP (Bellman-Ford)  |  Vertices: %d  |  Edges: %d\n\n",
            N, num_edges);

    // Allocate device memory
    int *d_row_offsets, *d_col_indices, *d_weights, *d_dist, *d_previous, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_previous, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    // Copy graph structure to device
    CUDA_CHECK(cudaMemcpy(d_row_offsets, row_offsets.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, col_indices.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t snapshot_acc, acc;
    int pass_iters = run_sssp_gpu(N, d_row_offsets, d_col_indices,
                                  d_weights, d_dist, d_previous, d_changed, source, block_size,
                                  &timer, &snapshot_acc, &acc);
    bench_timer_destroy(&timer);
    SsspWork work;

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification (compare GPU result with CPU reference)
    // -------------------------------------------------------------------
    {
        std::vector<int> h_gpu_dist(N);
        CUDA_CHECK(cudaMemcpy(h_gpu_dist.data(), d_dist, N * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> h_cpu_dist;
        int reference_iters = bellman_ford_cpu(
            N, row_offsets, col_indices, weights, h_cpu_dist, source, &work);
        if (pass_iters != reference_iters) {
            fprintf(stderr, "FAIL: BF iterations GPU=%d CPU=%d\n", pass_iters, reference_iters);
            verify_ok = false;
        }

        int errors = 0;
        for (int i = 0; i < N; ++i) {
            if (h_gpu_dist[i] != h_cpu_dist[i]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at vertex %d: GPU=%d CPU=%d\n",
                            i, h_gpu_dist[i], h_cpu_dist[i]);
                }
                errors++;
            }
        }
        if (errors > 0) {
            fprintf(stderr, "FAIL: %d mismatches in %d vertices\n", errors, N);
            verify_ok = false;
        } else if (verify_ok)
            fprintf(stderr, "PASS: 0 mismatches in %d vertices\n", N);
    }

    char params[512];
    snprintf(params, sizeof(params),
             "\"vertices\":%d,\"degree\":%d,\"block_size\":%d,"
             "\"input\":\"xorshift32_seed42_v1\",\"algorithm\":\"topology_snapshot_v1\","
             "\"verification\":\"full_cpu_distances_rounds_v2\","
             "\"graph_edges\":%d,\"iterations\":%d,\"edge_relaxations\":%llu",
             vertices, degree, block_size, num_edges, pass_iters, work.edge_relaxations);
    bench_emit_iter("sssp_snapshot_kernel", 0, &snapshot_acc, params);
    bench_emit_iter("bellman_ford_kernel", 0, &acc, params);
    bench_emit_summary(snapshot_acc.gpu_ms + acc.gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms (%d BF iterations, %llu edge attempts)\n",
            snapshot_acc.gpu_ms + acc.gpu_ms, pass_iters, work.edge_relaxations);

    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_previous));
    CUDA_CHECK(cudaFree(d_changed));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
