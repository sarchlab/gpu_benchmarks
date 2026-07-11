// lonestar_sssp.hip — Single-Source Shortest Path (Bellman-Ford) benchmark (HIP)
//
// Iterative edge relaxation on a random sparse graph stored in CSR format.
// All edges are relaxed in parallel each iteration; repeats until no changes.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
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
// Output (stdout): for the single timed run, one {"type":"kernel",...} JSON
// line for bellman_ford_kernel carrying the accumulated GPU event time
// (time_ms) and host time (host_time_ms) of all Bellman-Ford iterations
// until convergence, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. Medges/sec is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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

__global__ void bellman_ford_kernel(const int* __restrict__ row_offsets,
                                    const int* __restrict__ col_indices,
                                    const int* __restrict__ weights,
                                    int* __restrict__ dist,
                                    int* __restrict__ changed,
                                    int N) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    int d_u = dist[u];
    if (d_u == INT_MAX) return; // unreachable, skip

    int start = row_offsets[u];
    int end   = row_offsets[u + 1];

    for (int e = start; e < end; ++e) {
        int v = col_indices[e];
        int w = weights[e];
        int new_dist = d_u + w;
        // Atomic min relaxation
        int old = atomicMin(&dist[v], new_dist);
        if (new_dist < old) {
            *changed = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: Bellman-Ford
// ---------------------------------------------------------------------------

static int bellman_ford_cpu(int N, const std::vector<int>& row_offsets,
                             const std::vector<int>& col_indices,
                             const std::vector<int>& weights,
                             std::vector<int>& dist, int source) {
    dist.assign(N, INT_MAX);
    dist[source] = 0;
    int total_iters = 0;

    for (int iter = 0; iter < N; ++iter) {
        bool updated = false;
        for (int u = 0; u < N; ++u) {
            if (dist[u] == INT_MAX) continue;
            for (int e = row_offsets[u]; e < row_offsets[u + 1]; ++e) {
                int v = col_indices[e];
                int w = weights[e];
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    updated = true;
                }
            }
        }
        total_iters++;
        if (!updated) break;
    }
    return total_iters;
}

// ---------------------------------------------------------------------------
// GPU SSSP: run Bellman-Ford iterations
// ---------------------------------------------------------------------------

static int run_sssp_gpu(int N, int num_edges,
                         int* d_row_offsets, int* d_col_indices,
                         int* d_weights, int* d_dist, int* d_changed,
                         int source, int blkSize,
                         bench_timer_t* timer, bench_lap_t* acc) {
    bench_lap_reset(acc);

    // Reset per-pass state (untimed): distances back to INT_MAX / source = 0
    // so every pass performs the same convergence work.
    std::vector<int> h_dist(N, INT_MAX);
    h_dist[source] = 0;
    HIP_CHECK(hipMemcpy(d_dist, h_dist.data(), N * sizeof(int), hipMemcpyHostToDevice));

    int blockSize = blkSize;
    int gridSize  = (N + blockSize - 1) / blockSize;
    int max_iters = N; // upper bound
    int total_iters = 0;

    /* Cold-cache calibration: evict the LLC (and hold the clock high) after
     * the per-pass H2D re-init so the pass's first kernel launch runs cold.
     * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
    bench_pass_warm();

    for (int iter = 0; iter < max_iters; ++iter) {
        int h_changed = 0;
        HIP_CHECK(hipMemcpy(d_changed, &h_changed, sizeof(int), hipMemcpyHostToDevice));

        bench_lap_add(acc, bench_time_op(timer, [&]() {
            bellman_ford_kernel<<<dim3(gridSize), dim3(blockSize)>>>(
                d_row_offsets, d_col_indices, d_weights, d_dist, d_changed, N);
        }));

        HIP_CHECK(hipMemcpy(&h_changed, d_changed, sizeof(int), hipMemcpyDeviceToHost));
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

    // Generate CSR graph
    std::vector<int> row_offsets, col_indices, weights;
    generate_csr_graph(N, avg_deg, 42, row_offsets, col_indices, weights);
    int num_edges = row_offsets[N];

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "SSSP (Bellman-Ford)  |  Vertices: %d  |  Edges: %d\n\n",
            N, num_edges);

    // Allocate device memory
    int *d_row_offsets, *d_col_indices, *d_weights, *d_dist, *d_changed;
    HIP_CHECK(hipMalloc(&d_row_offsets, (N + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, num_edges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_weights, num_edges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_dist, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_changed, sizeof(int)));

    // Copy graph structure to device
    HIP_CHECK(hipMemcpy(d_row_offsets, row_offsets.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, col_indices.data(), num_edges * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, weights.data(), num_edges * sizeof(int), hipMemcpyHostToDevice));

    char params[160];
    snprintf(params, sizeof(params),
             "\"vertices\":%d,\"degree\":%d,\"block_size\":%d",
             vertices, degree, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;

    // -------------------------------------------------------------------
    // Timed pass — one record with the accumulated kernel time
    // -------------------------------------------------------------------
    double total_gpu_ms = 0.0;
    int pass_iters = run_sssp_gpu(N, num_edges, d_row_offsets, d_col_indices,
                                  d_weights, d_dist, d_changed, source, block_size,
                                  &timer, &acc);
    bench_emit_iter("bellman_ford_kernel", 0, &acc, params);
    total_gpu_ms += acc.gpu_ms;
    fprintf(stderr, "gpu = %.3f ms, host = %.3f ms (%d BF iterations)\n",
            acc.gpu_ms, acc.host_ms, pass_iters);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification (compare GPU result with CPU reference)
    // -------------------------------------------------------------------
    {
        std::vector<int> h_gpu_dist(N);
        HIP_CHECK(hipMemcpy(h_gpu_dist.data(), d_dist, N * sizeof(int), hipMemcpyDeviceToHost));

        std::vector<int> h_cpu_dist;
        bellman_ford_cpu(N, row_offsets, col_indices, weights, h_cpu_dist, source);

        int verify_n = (N < 1024) ? N : 1024;
        int errors = 0;
        for (int i = 0; i < verify_n; ++i) {
            if (h_gpu_dist[i] != h_cpu_dist[i]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at vertex %d: GPU=%d CPU=%d\n",
                            i, h_gpu_dist[i], h_cpu_dist[i]);
                }
                errors++;
            }
        }
        if (errors > 0) {
            fprintf(stderr, "FAIL: %d mismatches in first %d vertices\n",
                    errors, verify_n);
            verify_ok = false;
        } else
            fprintf(stderr, "PASS\n");
    }

    HIP_CHECK(hipFree(d_row_offsets));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_dist));
    HIP_CHECK(hipFree(d_changed));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
