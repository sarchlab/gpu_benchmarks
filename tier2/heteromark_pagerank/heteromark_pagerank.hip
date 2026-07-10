// heteromark_pagerank.hip — HeteroMark PageRank benchmark (HIP, self-contained)
//
// Iterative PageRank (power method) on a synthetic Erdős–Rényi graph in CSR
// format.  PR[v] = (1-d)/N + d * sum(PR[u]/degree[u]) for all u→v.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./heteromark_pagerank [--vertices N] [--pr-iterations P]
//
//   --vertices N       Number of vertices (default: 65536)
//   --pr-iterations P  PageRank iterations per run (default: 20)
//
// Environment variables (override argv):
//   BENCH_PARAM_vertices, BENCH_PARAM_pr_iterations, BENCH_PARAM_damping_factor
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line for
// pagerank_kernel carrying the accumulated GPU event time of all PageRank
// iterations in the pass plus the host wall-clock time, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Edges/sec is
// derived downstream from time_ms and params.
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
// Generate synthetic Erdős–Rényi graph in CSR format
// ---------------------------------------------------------------------------

static void generateCSR(int N, int avgDegree, uint32_t seed,
                         std::vector<int>& rowPtr,
                         std::vector<int>& colIdx,
                         std::vector<int>& outDegree) {
    // Build adjacency lists using deterministic PRNG
    // For each vertex u, randomly pick ~avgDegree target vertices
    std::vector<std::vector<int>> adj(N);
    uint32_t rng = seed;

    for (int u = 0; u < N; ++u) {
        int deg = avgDegree;
        for (int d = 0; d < deg; ++d) {
            int v = (int)(xorshift32(rng) % (uint32_t)N);
            adj[v].push_back(u);  // edge u→v stored in v's in-neighbor list
        }
    }

    // Out-degree of each vertex
    outDegree.resize(N);
    for (int u = 0; u < N; ++u) {
        outDegree[u] = avgDegree;  // each vertex has exactly avgDegree out-edges
    }

    // Build CSR for incoming edges (for PageRank gather)
    rowPtr.resize(N + 1);
    rowPtr[0] = 0;
    for (int v = 0; v < N; ++v) {
        rowPtr[v + 1] = rowPtr[v] + (int)adj[v].size();
    }

    int nnz = rowPtr[N];
    colIdx.resize(nnz);
    for (int v = 0; v < N; ++v) {
        int offset = rowPtr[v];
        for (int j = 0; j < (int)adj[v].size(); ++j) {
            colIdx[offset + j] = adj[v][j];
        }
    }
}

// ---------------------------------------------------------------------------
// PageRank kernel: one iteration
//   PR_new[v] = (1-d)/N + d * sum(PR_old[u]/degree[u]) for all u→v
// ---------------------------------------------------------------------------

__global__ void pagerank_kernel(const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const int* __restrict__ outDegree,
                                 const float* __restrict__ prOld,
                                 float* __restrict__ prNew,
                                 int N, float damping) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    float sum = 0.0f;
    int start = rowPtr[v];
    int end   = rowPtr[v + 1];

    for (int e = start; e < end; ++e) {
        int u = colIdx[e];
        sum += prOld[u] / (float)outDegree[u];
    }

    prNew[v] = (1.0f - damping) / (float)N + damping * sum;
}

// ---------------------------------------------------------------------------
// CPU reference PageRank for verification
// ---------------------------------------------------------------------------

static void pagerank_cpu(const std::vector<int>& rowPtr,
                          const std::vector<int>& colIdx,
                          const std::vector<int>& outDegree,
                          std::vector<float>& pr,
                          int N, int prIters, float damping) {
    std::vector<float> prNew(N);
    for (int i = 0; i < N; ++i) pr[i] = 1.0f / (float)N;

    for (int iter = 0; iter < prIters; ++iter) {
        for (int v = 0; v < N; ++v) {
            float sum = 0.0f;
            for (int e = rowPtr[v]; e < rowPtr[v + 1]; ++e) {
                int u = colIdx[e];
                sum += pr[u] / (float)outDegree[u];
            }
            prNew[v] = (1.0f - damping) / (float)N + damping * sum;
        }
        pr.swap(prNew);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static double parseDoubleParam(int argc, char** argv, const char* name, double defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            double v = atof(argv[i + 1]);
            return v;
        }
    }
    return defaultVal;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int vertices     = 65536;
    int pr_iterations = 20;
    double damping_factor = 0.85;

    // Command-line fallback
    vertices      = parseIntParam(argc, argv, "--vertices", vertices);
    pr_iterations = parseIntParam(argc, argv, "--pr-iterations", pr_iterations);
    damping_factor = parseDoubleParam(argc, argv, "--damping_factor", damping_factor);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_vertices");
    if (env_val) vertices = atoi(env_val);
    env_val = getenv("BENCH_PARAM_pr_iterations");
    if (env_val) pr_iterations = atoi(env_val);
    env_val = getenv("BENCH_PARAM_damping_factor");
    if (env_val) damping_factor = atof(env_val);
    int N       = vertices;
    int prIters = pr_iterations;
    float damping = (float)damping_factor;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Vertices: %d  |  PR iterations: %d\n\n",
            N, prIters);

    // Generate synthetic graph
    std::vector<int> rowPtr, colIdx, outDegree;
    generateCSR(N, 16, 42, rowPtr, colIdx, outDegree);
    int numEdges = rowPtr[N];
    fprintf(stderr, "Graph: %d vertices, %d edges (avg in-degree: %.1f)\n",
            N, numEdges, (double)numEdges / N);

    // Host PageRank arrays
    std::vector<float> h_pr(N, 1.0f / (float)N);
    std::vector<float> h_prNew(N);

    // Device allocations
    int *d_rowPtr, *d_colIdx, *d_outDegree;
    float *d_prOld, *d_prNew;

    HIP_CHECK(hipMalloc(&d_rowPtr, (N + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_colIdx, numEdges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_outDegree, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_prOld, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_prNew, N * sizeof(float)));

    // Copy graph to device (constant across runs)
    HIP_CHECK(hipMemcpy(d_rowPtr, rowPtr.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_colIdx, colIdx.data(), numEdges * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_outDegree, outDegree.data(), N * sizeof(int), hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    char params[160];
    snprintf(params, sizeof(params),
             "\"vertices\":%d,\"pr_iterations\":%d,\"damping_factor\":%.6f",
             vertices, pr_iterations, damping_factor);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_pagerank;

    // One full PageRank pass. The power method mutates the rank vectors, so
    // ranks are reset to 1/N at the start of every pass (untimed) — each pass
    // runs the same prIters iterations on the same initial state. Only the
    // kernel launches are inside the timed brackets; the pointer swap is host
    // bookkeeping outside them.
    auto run_pagerank_pass = [&]() {
        bench_lap_reset(&acc_pagerank);

        for (int i = 0; i < N; ++i) h_pr[i] = 1.0f / (float)N;
        HIP_CHECK(hipMemcpy(d_prOld, h_pr.data(), N * sizeof(float), hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int it = 0; it < prIters; ++it) {
            bench_lap_add(&acc_pagerank, bench_time_op(&timer, [&]() {
                pagerank_kernel<<<dim3(gridSize), dim3(blockSize)>>>(
                    d_rowPtr, d_colIdx, d_outDegree, d_prOld, d_prNew, N, damping);
            }));

            // Swap pointers
            float* tmp = d_prOld;
            d_prOld = d_prNew;
            d_prNew = tmp;
        }
    };

    run_pagerank_pass();
    double total_gpu_ms = 0.0;
    bench_emit_iter("pagerank_kernel", 0, &acc_pagerank, params);
    total_gpu_ms += acc_pagerank.gpu_ms;
    fprintf(stderr, "pagerank_kernel gpu = %.3f ms, host = %.3f ms\n",
            acc_pagerank.gpu_ms, acc_pagerank.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // Verification: compare GPU result against CPU reference
    // -------------------------------------------------------------------
    HIP_CHECK(hipMemcpy(h_pr.data(), d_prOld, N * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> refPr(N);
    pagerank_cpu(rowPtr, colIdx, outDegree, refPr, N, prIters, damping);

    int errors = 0;
    for (int v = 0; v < N; ++v) {
        float diff = fabsf(h_pr[v] - refPr[v]);
        float tol  = 1e-4f * fabsf(refPr[v]) + 1e-7f;
        if (diff > tol) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at v=%d: GPU=%.8f CPU=%.8f\n",
                        v, h_pr[v], refPr[v]);
            }
            errors++;
        }
    }
    if (errors > 0)
        fprintf(stderr, "FAIL: %d errors out of %d vertices\n", errors, N);
    else
        fprintf(stderr, "PASS\n");

    // Cleanup
    HIP_CHECK(hipFree(d_rowPtr));
    HIP_CHECK(hipFree(d_colIdx));
    HIP_CHECK(hipFree(d_outDegree));
    HIP_CHECK(hipFree(d_prOld));
    HIP_CHECK(hipFree(d_prNew));

    return 0;
}
