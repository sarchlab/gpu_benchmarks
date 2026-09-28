// graph_cc.hip — Graph Connected Components via label propagation (HIP)
//
// Computes connected components on a random undirected sparse graph using
// iterative label propagation. The graph is stored in CSR format. Each vertex
// starts with its own ID as label and iteratively updates to the minimum label
// among itself and its neighbors until convergence.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables:
//   BENCH_PARAM_num_vertices, BENCH_PARAM_avg_degree,
//   BENCH_PARAM_max_iterations, BENCH_PARAM_block_size
//
// Output (stdout): for the single timed run, one {"type":"kernel",...} JSON
// line for cc_propagate_kernel carrying the accumulated GPU event time
// (time_ms) and host time (host_time_ms) of all label-propagation iterations
// until convergence, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. MTEPS is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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
// Generate random undirected graph in CSR format
// ---------------------------------------------------------------------------

static void generate_csr_graph(int N, int avg_degree, uint32_t seed,
                                std::vector<int>& row_offsets,
                                std::vector<int>& col_indices) {
    uint32_t rng = seed;

    std::vector<std::vector<int>> adj(N);

    for (int u = 0; u < N; ++u) {
        int d = avg_degree / 2 + (int)(xorshift32(rng) % (avg_degree + 1));
        if (d < 1) d = 1;
        for (int e = 0; e < d; ++e) {
            int v = (int)(xorshift32(rng) % (uint32_t)N);
            if (v != u) {
                adj[u].push_back(v);
                adj[v].push_back(u);
            }
        }
    }

    row_offsets.resize(N + 1);
    row_offsets[0] = 0;
    for (int u = 0; u < N; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
        row_offsets[u + 1] = row_offsets[u] + (int)adj[u].size();
    }

    int total_edges = row_offsets[N];
    col_indices.resize(total_edges);

    for (int u = 0; u < N; ++u) {
        int offset = row_offsets[u];
        for (int i = 0; i < (int)adj[u].size(); ++i) {
            col_indices[offset + i] = adj[u][i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: CC label propagation
// ---------------------------------------------------------------------------

// Double-buffered (Jacobi) label propagation: read every label from labels_in
// and write the updated label to labels_out. The previous in-place version read
// and wrote one `labels` array, so a vertex could observe a neighbor's label
// from the current iteration or the previous one depending on block scheduling —
// making the iteration count (= measured work) vary run to run on identical
// input (rule 9). Reading only labels_in makes each iteration a pure function of
// the prior state, so convergence is deterministic. The fixed point (each vertex
// = min vertex id in its component) is unique, so results are unchanged.
__global__ void cc_propagate_kernel(const int* __restrict__ row_offsets,
                                     const int* __restrict__ col_indices,
                                     const int* __restrict__ labels_in,
                                     int* __restrict__ labels_out,
                                     int* __restrict__ changed,
                                     int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    int my_label = labels_in[v];
    int new_label = my_label;

    int start = row_offsets[v];
    int end   = row_offsets[v + 1];

    for (int e = start; e < end; ++e) {
        int u = col_indices[e];
        int neighbor_label = labels_in[u];
        if (neighbor_label < new_label) {
            new_label = neighbor_label;
        }
    }

    labels_out[v] = new_label;      // always write — Jacobi update
    if (new_label < my_label) {
        *changed = 1;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: CC via label propagation
// ---------------------------------------------------------------------------

static void cc_cpu(int N, int max_iters,
                   const std::vector<int>& row_offsets,
                   const std::vector<int>& col_indices,
                   std::vector<int>& labels) {
    labels.resize(N);
    for (int i = 0; i < N; ++i) labels[i] = i;

    // Jacobi (double-buffered) propagation, matching the GPU kernel: every
    // vertex reads neighbor labels from the previous iteration (`labels`) and
    // writes to `next`, then the buffers swap. Using the same semantics as the
    // device makes this a faithful reference (identical per-iteration state, not
    // just the same final fixed point).
    std::vector<int> next(labels);
    for (int iter = 0; iter < max_iters; ++iter) {
        bool any_changed = false;
        for (int v = 0; v < N; ++v) {
            int new_label = labels[v];
            for (int e = row_offsets[v]; e < row_offsets[v + 1]; ++e) {
                int u = col_indices[e];
                if (labels[u] < new_label) {
                    new_label = labels[u];
                }
            }
            next[v] = new_label;
            if (new_label < labels[v]) any_changed = true;
        }
        labels.swap(next);
        if (!any_changed) break;
    }
}

// ---------------------------------------------------------------------------
// GPU CC: one full pass — reset labels (untimed), then propagate until
// convergence or max_iters. Every kernel launch is bracketed with hipEvents;
// the accumulated GPU/host time of all iterations lands in *acc.
// ---------------------------------------------------------------------------

static int run_cc_gpu(int N, int max_iters, int block_size,
                      int* d_row_offsets, int* d_col_indices,
                      int* d_labels, int* d_labels_scratch, int* d_changed,
                      bench_timer_t* timer, bench_lap_t* acc) {
    bench_lap_reset(acc);

    // Reset per-pass state: labels back to vertex IDs so every pass performs
    // the same convergence work regardless of what earlier passes left behind.
    std::vector<int> h_init(N);
    for (int i = 0; i < N; ++i) h_init[i] = i;
    HIP_CHECK(hipMemcpy(d_labels, h_init.data(), N * sizeof(int), hipMemcpyHostToDevice));

    int gridSize = (N + block_size - 1) / block_size;
    int h_changed = 0;
    int iters_done = 0;

    /* Cold-cache calibration: evict the LLC (and hold the clock high) after
     * the per-pass H2D re-init so the pass's first kernel launch runs cold.
     * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
    bench_pass_warm();

    // Ping-pong: read `cur`, write `nxt`, then swap. `cur` starts at d_labels.
    int* cur = d_labels;
    int* nxt = d_labels_scratch;
    for (int iter = 0; iter < max_iters; ++iter) {
        h_changed = 0;
        HIP_CHECK(hipMemcpy(d_changed, &h_changed, sizeof(int), hipMemcpyHostToDevice));

        bench_lap_add(acc, bench_time_op(timer, [&]() {
            cc_propagate_kernel<<<dim3(gridSize), dim3(block_size)>>>(
                d_row_offsets, d_col_indices, cur, nxt, d_changed, N);
        }));

        HIP_CHECK(hipMemcpy(&h_changed, d_changed, sizeof(int), hipMemcpyDeviceToHost));
        int* tmp = cur; cur = nxt; nxt = tmp;   // updated labels are now in cur
        iters_done = iter + 1;
        if (h_changed == 0) break;
    }

    // Leave the final labels in d_labels for the caller (untimed D2D copy only
    // when the last write landed in the scratch buffer).
    if (cur != d_labels) {
        HIP_CHECK(hipMemcpy(d_labels, cur, N * sizeof(int), hipMemcpyDeviceToDevice));
    }

    return iters_done;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int num_vertices   = 100000;
    int avg_degree     = 16;
    int max_iterations = 100;
    int block_size     = 256;

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_num_vertices");
    if (env_val) num_vertices = atoi(env_val);
    env_val = getenv("BENCH_PARAM_avg_degree");
    if (env_val) avg_degree = atoi(env_val);
    env_val = getenv("BENCH_PARAM_max_iterations");
    if (env_val) max_iterations = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int N = num_vertices;

    // Generate CSR graph
    std::vector<int> row_offsets, col_indices;
    generate_csr_graph(N, avg_degree, 42, row_offsets, col_indices);
    int num_edges = row_offsets[N];

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Connected Components (label propagation)  |  "
            "Vertices: %d  |  Edges: %d  |  Avg degree: %d  |  "
            "Max iterations: %d\n\n",
            N, num_edges, avg_degree, max_iterations);

    // Allocate device memory
    int *d_row_offsets, *d_col_indices, *d_labels, *d_labels2, *d_changed;
    HIP_CHECK(hipMalloc(&d_row_offsets, (N + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, num_edges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_labels,  N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_labels2, N * sizeof(int)));   // ping-pong scratch
    HIP_CHECK(hipMalloc(&d_changed, sizeof(int)));

    HIP_CHECK(hipMemcpy(d_row_offsets, row_offsets.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, col_indices.data(), num_edges * sizeof(int), hipMemcpyHostToDevice));

    char params[192];
    snprintf(params, sizeof(params),
             "\"num_vertices\":%d,\"avg_degree\":%d,\"max_iterations\":%d,"
             "\"block_size\":%d,\"verification\":\"full_cpu_v1\"",
             num_vertices, avg_degree, max_iterations, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;

    // -------------------------------------------------------------------
    // Timed pass — one record with the accumulated kernel time
    // -------------------------------------------------------------------
    double total_gpu_ms = 0.0;
    int cc_iters = run_cc_gpu(N, max_iterations, block_size,
                              d_row_offsets, d_col_indices, d_labels, d_labels2,
                              d_changed, &timer, &acc);
    bench_emit_iter("cc_propagate_kernel", 0, &acc, params);
    total_gpu_ms += acc.gpu_ms;
    fprintf(stderr, "gpu = %.3f ms, host = %.3f ms (%d CC iterations)\n",
            acc.gpu_ms, acc.host_ms, cc_iters);

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------
    // The timed pass already left the final labels in d_labels, so we read
    // them directly — no second run_cc_gpu() (which would also fire a second
    // untimed warmer and re-solve). Repetition across processes is the
    // harness's job.
    {
        std::vector<int> h_gpu_labels(N);
        HIP_CHECK(hipMemcpy(h_gpu_labels.data(), d_labels, N * sizeof(int), hipMemcpyDeviceToHost));

        std::vector<int> h_cpu_labels;
        cc_cpu(N, max_iterations, row_offsets, col_indices, h_cpu_labels);

        int errors = 0;

        for (int i = 0; i < N; ++i) {
            if (h_gpu_labels[i] != h_cpu_labels[i]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at vertex %d: GPU=%d CPU=%d\n",
                            i, h_gpu_labels[i], h_cpu_labels[i]);
                }
                errors++;
            }
        }

        verify_ok = errors == 0;
        fprintf(stderr, "CC verification: %s (%d vertices, %d mismatches)\n",
                verify_ok ? "PASS" : "FAIL", N, errors);
    }

    bench_timer_destroy(&timer);
    HIP_CHECK(hipFree(d_row_offsets));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_labels));
    HIP_CHECK(hipFree(d_labels2));
    HIP_CHECK(hipFree(d_changed));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
