// graph_pr.hip — Graph PageRank (iterative, pull-based) benchmark (HIP)
//
// Pull-based PageRank with CSC format on a random sparse graph.
// Each thread computes the new rank for one vertex by pulling from incoming
// neighbors.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./graph_pr [--vertices N] [--degree D] [--pr-iterations P]
//
//   --vertices N       Number of vertices (default: 131072)
//   --degree D         Average degree per vertex (default: 16)
//   --pr-iterations P  PageRank iterations per run (default: 5)
//
// Environment variables (override argv):
//   BENCH_PARAM_vertices, BENCH_PARAM_degree, BENCH_PARAM_pr_iterations
//
// Output (stdout): for the single timed run, one {"type":"kernel",...} JSON
// line for pagerank_kernel carrying the accumulated GPU event time of all PR
// iterations within the run plus host wall-clock time, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. GTEPS is derived
// downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

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
// Generate random graph in CSC format (for pull-based PageRank)
// Also computes out-degree for each vertex.
// ---------------------------------------------------------------------------

static void generate_csc_graph(int N, int avg_degree, uint32_t seed,
                                std::vector<int>& col_offsets,
                                std::vector<int>& row_indices,
                                std::vector<int>& out_degree) {
    uint32_t rng = seed;

    // First generate edges as (src, dst) pairs
    // We generate avg_degree outgoing edges per vertex
    std::vector<std::vector<int>> incoming(N);
    out_degree.assign(N, 0);

    for (int u = 0; u < N; ++u) {
        int d = avg_degree / 2 + (int)(xorshift32(rng) % (avg_degree + 1));
        if (d < 1) d = 1;
        out_degree[u] = d;
        for (int e = 0; e < d; ++e) {
            int v = (int)(xorshift32(rng) % (uint32_t)N);
            incoming[v].push_back(u);
        }
    }

    // Build CSC from incoming lists
    col_offsets.resize(N + 1);
    col_offsets[0] = 0;
    for (int v = 0; v < N; ++v)
        col_offsets[v + 1] = col_offsets[v] + (int)incoming[v].size();

    int total_edges = col_offsets[N];
    row_indices.resize(total_edges);

    for (int v = 0; v < N; ++v) {
        int offset = col_offsets[v];
        for (int i = 0; i < (int)incoming[v].size(); ++i) {
            row_indices[offset + i] = incoming[v][i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Pull-based PageRank iteration
// Each thread computes the new rank for one vertex
// ---------------------------------------------------------------------------

__global__ void pagerank_kernel(const int* __restrict__ col_offsets,
                                 const int* __restrict__ row_indices,
                                 const int* __restrict__ out_degree,
                                 const float* __restrict__ rank_in,
                                 float* __restrict__ rank_out,
                                 float damping,
                                 float base_rank,
                                 int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    int start = col_offsets[v];
    int end   = col_offsets[v + 1];

    float sum = 0.0f;
    for (int e = start; e < end; ++e) {
        int u = row_indices[e];
        int deg = out_degree[u];
        if (deg > 0)
            sum += rank_in[u] / (float)deg;
    }

    rank_out[v] = base_rank + damping * sum;
}

// ---------------------------------------------------------------------------
// CPU reference: PageRank
// ---------------------------------------------------------------------------

static void pagerank_cpu(int N, int pr_iters, float damping,
                          const std::vector<int>& col_offsets,
                          const std::vector<int>& row_indices,
                          const std::vector<int>& out_degree,
                          std::vector<float>& rank_out) {
    float base_rank = (1.0f - damping) / (float)N;
    std::vector<float> rank_a(N, 1.0f / (float)N);
    std::vector<float> rank_b(N);

    float* cur = rank_a.data();
    float* nxt = rank_b.data();

    for (int iter = 0; iter < pr_iters; ++iter) {
        for (int v = 0; v < N; ++v) {
            float sum = 0.0f;
            for (int e = col_offsets[v]; e < col_offsets[v + 1]; ++e) {
                int u = row_indices[e];
                if (out_degree[u] > 0)
                    sum += cur[u] / (float)out_degree[u];
            }
            nxt[v] = base_rank + damping * sum;
        }
        float* tmp = cur; cur = nxt; nxt = tmp;
    }

    rank_out.assign(cur, cur + N);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int vertices      = 131072;
    int degree        = 16;
    int pr_iterations = 5;   // default from params.json

    // Command-line fallback
    vertices      = parseIntParam(argc, argv, "--vertices", vertices);
    degree        = parseIntParam(argc, argv, "--degree", degree);
    pr_iterations = parseIntParam(argc, argv, "--pr-iterations", pr_iterations);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_vertices");
    if (env_val) vertices = atoi(env_val);
    env_val = getenv("BENCH_PARAM_degree");
    if (env_val) degree = atoi(env_val);
    env_val = getenv("BENCH_PARAM_pr_iterations");
    if (env_val) pr_iterations = atoi(env_val);

    int N         = vertices;
    int avg_deg   = degree;
    int pr_iters  = pr_iterations;
    float damping = 0.85f;

    // Generate CSC graph
    std::vector<int> col_offsets, row_indices, out_degree;
    generate_csc_graph(N, avg_deg, 42, col_offsets, row_indices, out_degree);
    int num_edges = col_offsets[N];

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "PageRank (pull-based)  |  Vertices: %d  |  Edges: %d  |  "
            "PR iterations: %d  |  Damping: %.2f\n\n",
            N, num_edges, pr_iters, damping);

    // Allocate device memory
    int *d_col_offsets, *d_row_indices, *d_out_degree;
    float *d_rank_a, *d_rank_b;
    HIP_CHECK(hipMalloc(&d_col_offsets, (N + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_row_indices, num_edges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out_degree, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_rank_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rank_b, N * sizeof(float)));

    // Copy graph structure to device
    HIP_CHECK(hipMemcpy(d_col_offsets, col_offsets.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_indices, row_indices.data(), num_edges * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out_degree, out_degree.data(), N * sizeof(int), hipMemcpyHostToDevice));

    char params[160];
    snprintf(params, sizeof(params),
             "\"vertices\":%d,\"degree\":%d,\"pr_iterations\":%d",
             vertices, degree, pr_iterations);

    float base_rank = (1.0f - damping) / (float)N;
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    std::vector<float> h_init(N, 1.0f / (float)N);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_pr;

    // One full PageRank pass: rank vector is re-initialized at the start of
    // every pass (untimed), so each pass runs exactly pr_iters identical
    // iterations regardless of what earlier passes left in the rank buffers.
    auto run_pr_pass = [&]() {
        bench_lap_reset(&acc_pr);

        HIP_CHECK(hipMemcpy(d_rank_a, h_init.data(), N * sizeof(float),
                            hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        float* cur = d_rank_a;
        float* nxt = d_rank_b;
        for (int iter = 0; iter < pr_iters; ++iter) {
            bench_lap_add(&acc_pr, bench_time_op(&timer, [&]() {
                pagerank_kernel<<<dim3(gridSize), dim3(blockSize)>>>(
                    d_col_offsets, d_row_indices, d_out_degree, cur, nxt,
                    damping, base_rank, N);
            }));
            float* tmp = cur; cur = nxt; nxt = tmp;
        }

        // Final result lands in d_rank_a for verification (untimed)
        if (cur != d_rank_a) {
            HIP_CHECK(hipMemcpy(d_rank_a, cur, N * sizeof(float),
                                hipMemcpyDeviceToDevice));
        }
    };

    double total_gpu_ms = 0.0;
    run_pr_pass();
    bench_emit_iter("pagerank_kernel", 0, &acc_pr, params);
    total_gpu_ms += acc_pr.gpu_ms;
    fprintf(stderr, "pagerank_kernel gpu = %.3f ms, host = %.3f ms\n",
            acc_pr.gpu_ms, acc_pr.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification (compare GPU result with CPU reference)
    // -------------------------------------------------------------------
    {
        std::vector<float> h_gpu_rank(N);
        HIP_CHECK(hipMemcpy(h_gpu_rank.data(), d_rank_a, N * sizeof(float), hipMemcpyDeviceToHost));

        std::vector<float> h_cpu_rank;
        pagerank_cpu(N, pr_iters, damping, col_offsets, row_indices, out_degree, h_cpu_rank);

        int verify_n = (N < 1024) ? N : 1024;
        int errors = 0;
        float max_err = 0.0f;
        // Ranks have magnitude ~1/N, so an absolute tolerance would be
        // vacuous at large N. Use a scale-aware per-element check with a
        // 1e-2 relative term (covers float accumulation-order differences
        // between the CPU and GPU gathers over high-degree vertices).
        const float rank_floor = 1.0f / (float)N;
        for (int i = 0; i < verify_n; ++i) {
            float err = fabsf(h_gpu_rank[i] - h_cpu_rank[i]);
            if (err > max_err) max_err = err;
            float scale = fabsf(h_cpu_rank[i]);
            if (scale < rank_floor) scale = rank_floor;
            if (!std::isfinite(h_gpu_rank[i]) || !std::isfinite(h_cpu_rank[i]) ||
                err > 1e-7f + 1e-2f * scale) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at vertex %d: GPU=%.6f CPU=%.6f (err=%.6f)\n",
                            i, h_gpu_rank[i], h_cpu_rank[i], err);
                }
                errors++;
            }
        }
        fprintf(stderr, "Max error (first %d vertices): %.6e\n", verify_n, max_err);

        // Mass conservation: the generator guarantees min out-degree 1 (no
        // dangling nodes), so the ranks form a probability distribution and
        // must sum to 1.
        double rank_sum = 0.0;
        for (int i = 0; i < N; ++i) rank_sum += h_gpu_rank[i];
        bool mass_ok = fabs(rank_sum - 1.0) <= 1e-3;
        if (!mass_ok)
            fprintf(stderr, "Mass conservation violated: sum(rank) = %.6f\n",
                    rank_sum);

        if (errors > 0 || !mass_ok) {
            fprintf(stderr, "FAIL: %d mismatches in first %d vertices%s\n",
                    errors, verify_n,
                    mass_ok ? "" : ", rank mass not conserved");
            verify_ok = false;
        } else
            fprintf(stderr, "PASS\n");
    }

    HIP_CHECK(hipFree(d_col_offsets));
    HIP_CHECK(hipFree(d_row_indices));
    HIP_CHECK(hipFree(d_out_degree));
    HIP_CHECK(hipFree(d_rank_a));
    HIP_CHECK(hipFree(d_rank_b));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
