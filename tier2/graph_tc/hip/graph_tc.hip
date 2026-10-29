// graph_tc.hip — Triangle Counting benchmark (HIP)
//
// Counts triangles in an undirected graph using sorted adjacency list
// intersection. Each thread processes one edge (u, v) and counts common
// neighbors of u and v via a merge-based intersection.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./graph_tc [N=16384] [D=32]
//
//   N=<vertices>     Number of vertices (default: 16384)
//   D=<avg_degree>   Average degree (default: 32)
//
// Environment variables (override argv):
//   BENCH_PARAM_vertices, BENCH_PARAM_avg_degree, BENCH_PARAM_block_size
//   BENCH_PARAM_verify=false|0|skip skips CPU verification (debug knob, not a
//   declared parameter; records identify the effective verification policy)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// carrying the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the SUM of all iterations' GPU event times.
// Triangles/sec is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../../common_hip/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "graph_tc_host.h"

using graph_tc_host::generate_graph;
using graph_tc_host::triangle_count_cpu;

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing (key=value style)
// ---------------------------------------------------------------------------

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    size_t nlen = strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], name, nlen) == 0) {
            int v = atoi(argv[i] + nlen);
            if (v > 0) return v;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Kernel: Count triangles via sorted-list intersection
// Each thread handles one edge (u, v) with u < v, and counts ALL common
// neighbors of u and v. Each triangle is therefore counted once per each of
// its three edges; the host divides the summed counts by 3.
// ---------------------------------------------------------------------------

__global__ void triangle_count_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    long long* __restrict__ edge_counts,
    int num_edges,
    const int* __restrict__ edge_src,
    const int* __restrict__ edge_dst)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;

    int u = edge_src[eid];
    int v = edge_dst[eid];

    // Intersect sorted neighbor lists of u and v
    int u_start = row_ptr[u], u_end = row_ptr[u + 1];
    int v_start = row_ptr[v], v_end = row_ptr[v + 1];

    long long count = 0;
    int i = u_start, j = v_start;
    while (i < u_end && j < v_end) {
        int nu = col_idx[i];
        int nv = col_idx[j];
        if (nu == nv) {
            count++;
            i++;
            j++;
        } else if (nu < nv) {
            i++;
        } else {
            j++;
        }
    }

    edge_counts[eid] = count;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int vertices   = 16384;
    int avg_degree = 32;
    int block_size = 256;

    // Command-line fallback
    vertices   = parseIntParam(argc, argv, "N=", vertices);
    avg_degree = parseIntParam(argc, argv, "D=", avg_degree);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_vertices");
    if (env_val) vertices = atoi(env_val);
    env_val = getenv("BENCH_PARAM_avg_degree");
    if (env_val) avg_degree = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    // Debug knob (not a declared parameter): BENCH_PARAM_verify=false|0|skip
    // skips the CPU verification pass (expensive at large sizes); default runs it.
    const char* verify_env = getenv("BENCH_PARAM_verify");
    bool run_verify = !(verify_env &&
                        (strcmp(verify_env, "false") == 0 ||
                         strcmp(verify_env, "0") == 0 ||
                         strcmp(verify_env, "skip") == 0));
    int N = vertices;
    int D = avg_degree;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Triangle Counting  |  Vertices: %d  |  Avg Degree: %d\n\n",
            N, D);

    // Generate graph
    fprintf(stderr, "Generating random graph...\n");
    std::vector<int> row_ptr, col_idx;
    generate_graph(N, D, row_ptr, col_idx);
    int num_nnz = (int)col_idx.size();
    fprintf(stderr, "Graph: %d vertices, %d directed edges (CSR nnz)\n", N, num_nnz);

    // Build edge list for directed edges (u < v) only
    std::vector<int> edge_src, edge_dst;
    for (int u = 0; u < N; ++u) {
        for (int idx = row_ptr[u]; idx < row_ptr[u + 1]; ++idx) {
            int v = col_idx[idx];
            if (v > u) {
                edge_src.push_back(u);
                edge_dst.push_back(v);
            }
        }
    }
    int num_edges = (int)edge_src.size();
    fprintf(stderr, "Undirected edges (u<v): %d\n", num_edges);

    // Allocate device memory
    int* d_row_ptr;
    int* d_col_idx;
    int* d_edge_src;
    int* d_edge_dst;
    long long* d_edge_counts;

    HIP_CHECK(hipMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx, num_nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_edge_src, num_edges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_edge_dst, num_edges * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_edge_counts, num_edges * sizeof(long long)));

    HIP_CHECK(hipMemcpy(d_row_ptr, row_ptr.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, col_idx.data(), num_nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_edge_src, edge_src.data(), num_edges * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_edge_dst, edge_dst.data(), num_edges * sizeof(int), hipMemcpyHostToDevice));

    int blockSize = block_size;
    int gridSize = (num_edges + blockSize - 1) / blockSize;

    char params[192];
    snprintf(params, sizeof(params),
             "\"vertices\":%d,\"avg_degree\":%d,\"block_size\":%d,\"verification\":\"%s\"",
             vertices, avg_degree, block_size,
             run_verify ? "raw_triangle_total_cpu_v1" : "disabled");

    // Single timed launch (repetition is the harness's job). The kernel
    // reads only the immutable graph arrays and fully overwrites every
    // edge_counts element on each launch, so no state reset is needed
    // between executions.
    double total_gpu_ms =
        bench_run_once("triangle_count_kernel", params, [&]() {
            triangle_count_kernel<<<dim3(gridSize), dim3(blockSize)>>>(
                d_row_ptr, d_col_idx, d_edge_counts, num_edges, d_edge_src, d_edge_dst);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Sum up triangle counts from GPU
    std::vector<long long> h_edge_counts(num_edges);
    HIP_CHECK(hipMemcpy(h_edge_counts.data(), d_edge_counts,
                         num_edges * sizeof(long long), hipMemcpyDeviceToHost));

    long long gpu_total = 0;
    for (int i = 0; i < num_edges; ++i) {
        gpu_total += h_edge_counts[i];
    }
    // Each triangle (a,b,c) with a<b<c: edge (a,b) counts c, edge (a,c) counts b,
    // edge (b,c) counts a. So each triangle is counted 3 times. But common neighbors
    // include ALL neighbors, not just those forming ordered triples.
    // Actually for edge (u,v) with u<v, we count ALL common neighbors w.
    // w could be < u, between u and v, or > v.
    // Triangle (w,u,v) is counted on edge (u,v) for neighbor w.
    // The same triangle is also counted on edge (w,u) [if w<u] for neighbor v, and on edge (w,v) for neighbor u.
    // So each triangle is counted exactly 3 times across all edges.
    long long num_triangles = gpu_total / 3;

    // Human-readable output to stderr
    fprintf(stderr, "Triangles found:  %lld\n", num_triangles);

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification: compare against CPU reference
    // (skipped when BENCH_PARAM_verify=false|0|skip)
    // -------------------------------------------------------------------
    if (run_verify) {
        fprintf(stderr, "Running CPU verification...\n");
        long long cpu_raw = triangle_count_cpu(row_ptr, col_idx, N);
        long long cpu_triangles = cpu_raw / 3;

        // A correct raw total has exactly three contributions per triangle.
        // Comparing quotients hides erroneous totals cpu_raw+1 and cpu_raw+2.
        if (gpu_total == cpu_raw) {
            fprintf(stderr, "PASS (CPU=%lld, GPU=%lld triangles)\n",
                    cpu_triangles, num_triangles);
        } else {
            fprintf(stderr, "FAIL (CPU=%lld, GPU=%lld triangles, raw GPU=%lld, raw CPU=%lld)\n",
                    cpu_triangles, num_triangles, gpu_total, cpu_raw);
            verify_ok = false;
        }
    }

    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_edge_src));
    HIP_CHECK(hipFree(d_edge_dst));
    HIP_CHECK(hipFree(d_edge_counts));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
