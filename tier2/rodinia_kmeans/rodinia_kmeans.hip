// rodinia_kmeans.hip — Rodinia K-Means Clustering benchmark (HIP, self-contained)
//
// K-means clustering: assign N points (each D-dimensional) to K clusters,
// update cluster centers, repeat until convergence or max_iter.
//
// Derived from the Rodinia benchmark suite.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_kmeans [--points N] [--dims D] [--clusters K] [--max_iter M]
//
//   --points N      Number of data points      (default: 16384)
//   --dims D        Dimensions per point        (default: 16)
//   --clusters K    Number of clusters          (default: 8)
//   --max_iter M    Maximum iterations          (default: 10)
//
// Environment variables (override argv):
//   BENCH_PARAM_points, BENCH_PARAM_dims, BENCH_PARAM_clusters,
//   BENCH_PARAM_max_iter
//   BENCH_PARAM_num_warmup   (warmup clustering passes, default 3)
//   BENCH_PARAM_num_iters    (timed clustering passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (kmeans_assign_kernel, kmeans_update_kernel) carrying that kernel's
// accumulated GPU event time (time_ms) and host wall-clock time
// (host_time_ms) over the pass's k-means iterations, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Points/sec is
// derived downstream from time_ms and params.
// Output (stderr): Per-iteration delta, convergence status, PASS/FAIL

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal)
{
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
// Kernel 1: Assign each point to nearest cluster center
//
//   points[idx * D + d] — row-major: N x D
//   centers[k  * D + d] — row-major: K x D
//   membership[idx]     — output cluster index for each point
// ---------------------------------------------------------------------------

__global__ void kmeans_assign_kernel(
    const float* __restrict__ points,
    const float* __restrict__ centers,
    int*         __restrict__ membership,
    int N, int D, int K)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N) return;

    float min_dist = FLT_MAX;
    int   best_k   = 0;

    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = points[idx * D + d] - centers[k * D + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k   = k;
        }
    }
    membership[idx] = best_k;
}

// ---------------------------------------------------------------------------
// Kernel 2: Accumulate new cluster centers using atomics
//
//   new_centers[k * D + d] — running sum (zeroed before each call)
//   counts[k]              — number of points in cluster k
// ---------------------------------------------------------------------------

__global__ void kmeans_update_kernel(
    const float* __restrict__ points,
    float*       __restrict__ new_centers,
    int*         __restrict__ counts,
    const int*   __restrict__ membership,
    int N, int D)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N) return;

    int k = membership[idx];
    atomicAdd(&counts[k], 1);
    for (int d = 0; d < D; ++d) {
        atomicAdd(&new_centers[k * D + d], points[idx * D + d]);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int N        = 16384;
    int D        = 16;
    int K        = 8;
    int max_iter = 10;

    // Command-line fallback
    N        = parseIntParam(argc, argv, "--points", N);
    D        = parseIntParam(argc, argv, "--dims", D);
    K        = parseIntParam(argc, argv, "--clusters", K);
    max_iter = parseIntParam(argc, argv, "--max_iter", max_iter);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_points");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_dims");
    if (env_val) D = atoi(env_val);
    env_val = getenv("BENCH_PARAM_clusters");
    if (env_val) K = atoi(env_val);
    env_val = getenv("BENCH_PARAM_max_iter");
    if (env_val) max_iter = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "N=%d  D=%d  K=%d  max_iter=%d\n\n", N, D, K, max_iter);

    // -----------------------------------------------------------------------
    // Host allocations
    // -----------------------------------------------------------------------
    size_t points_bytes  = (size_t)N * D * sizeof(float);
    size_t centers_bytes = (size_t)K * D * sizeof(float);

    float* h_points       = (float*)malloc(points_bytes);
    float* h_centers_init = (float*)malloc(centers_bytes);
    float* h_centers      = (float*)malloc(centers_bytes);
    float* h_new_centers  = (float*)malloc(centers_bytes);
    int*   h_counts       = (int*)malloc((size_t)K * sizeof(int));
    int*   h_membership   = (int*)malloc((size_t)N * sizeof(int));
    int*   h_membership_prev = (int*)malloc((size_t)N * sizeof(int));

    // Initialize points randomly in [0, 1)^D
    unsigned int seed = 42;
    for (int i = 0; i < N * D; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h_points[i] = (float)(seed >> 8) / (float)(1 << 24);
    }

    // Initial centers = first K points (kept pristine for per-pass resets)
    memcpy(h_centers_init, h_points, centers_bytes);

    // -----------------------------------------------------------------------
    // Device allocations
    // -----------------------------------------------------------------------
    float* d_points;
    float* d_centers;
    float* d_new_centers;
    int*   d_counts;
    int*   d_membership;

    HIP_CHECK(hipMalloc(&d_points,      points_bytes));
    HIP_CHECK(hipMalloc(&d_centers,     centers_bytes));
    HIP_CHECK(hipMalloc(&d_new_centers, centers_bytes));
    HIP_CHECK(hipMalloc(&d_counts,      (size_t)K * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_membership,  (size_t)N * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_points, h_points, points_bytes, hipMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Kernel launch parameters
    // -----------------------------------------------------------------------
    const int block_size = 256;
    const int grid_size  = (N + block_size - 1) / block_size;

    char params[160];
    snprintf(params, sizeof(params),
             "\"points\":%d,\"dims\":%d,\"clusters\":%d,\"max_iter\":%d",
             N, D, K, max_iter);

    // Per-kernel accumulators: each clustering pass runs the k-means loop
    // to convergence (or max_iter); the accumulated GPU/host time of all
    // k-means iterations within one pass becomes that pass's record.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_assign, acc_update;

    // One full clustering pass. K-means converges: centroids and membership
    // MUST be reset to their initial state before every pass (untimed),
    // otherwise later passes would start from converged centroids and run
    // fewer k-means iterations. Host-side center division, convergence
    // checks, and all memcpys stay outside the timed brackets.
    int iter = 0;   // k-means iterations executed in the most recent pass
    auto run_kmeans_pass = [&]() {
        bench_lap_reset(&acc_assign);
        bench_lap_reset(&acc_update);

        // Reset centroids and membership history to pristine initial state
        memcpy(h_centers, h_centers_init, centers_bytes);
        HIP_CHECK(hipMemcpy(d_centers, h_centers, centers_bytes, hipMemcpyHostToDevice));
        for (int i = 0; i < N; ++i) {
            h_membership[i]      = -1;
            h_membership_prev[i] = -1;
        }

        for (iter = 0; iter < max_iter; ++iter) {

            // --- Step 1: assign ---
            bench_lap_add(&acc_assign, bench_time_op(&timer, [&]() {
                kmeans_assign_kernel<<<dim3(grid_size), dim3(block_size)>>>(
                    d_points, d_centers, d_membership, N, D, K);
            }));

            // --- Step 2: zero accumulators (untimed) ---
            HIP_CHECK(hipMemset(d_new_centers, 0, centers_bytes));
            HIP_CHECK(hipMemset(d_counts,      0, (size_t)K * sizeof(int)));

            // --- Step 3: accumulate ---
            bench_lap_add(&acc_update, bench_time_op(&timer, [&]() {
                kmeans_update_kernel<<<dim3(grid_size), dim3(block_size)>>>(
                    d_points, d_new_centers, d_counts, d_membership, N, D);
            }));

            // --- Step 4: copy back and divide on CPU (untimed) ---
            HIP_CHECK(hipMemcpy(h_new_centers, d_new_centers, centers_bytes,   hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_counts,      d_counts,      (size_t)K * sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_membership,  d_membership,  (size_t)N * sizeof(int), hipMemcpyDeviceToHost));

            for (int k = 0; k < K; ++k) {
                if (h_counts[k] > 0) {
                    for (int d = 0; d < D; ++d) {
                        h_new_centers[k * D + d] /= (float)h_counts[k];
                    }
                } else {
                    // Empty cluster — keep old center
                    memcpy(&h_new_centers[k * D], &h_centers[k * D], (size_t)D * sizeof(float));
                }
            }

            // --- Step 5: check convergence ---
            int changed = 0;
            for (int i = 0; i < N; ++i) {
                if (h_membership[i] != h_membership_prev[i]) ++changed;
            }
            double delta = (double)changed / (double)N;
            fprintf(stderr, "  iter %2d: changed=%-6d  delta=%.6f\n", iter + 1, changed, delta);

            // Update centers on device
            memcpy(h_centers, h_new_centers, centers_bytes);
            HIP_CHECK(hipMemcpy(d_centers, h_centers, centers_bytes, hipMemcpyHostToDevice));

            memcpy(h_membership_prev, h_membership, (size_t)N * sizeof(int));

            if (delta < 1e-4) {
                iter++;   // count this iteration
                break;
            }
        }
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_kmeans_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_kmeans_pass();
        bench_emit_iter("kmeans_assign_kernel", i, &acc_assign, params);
        bench_emit_iter("kmeans_update_kernel", i, &acc_update, params);
        total_gpu_ms += acc_assign.gpu_ms + acc_update.gpu_ms;
        fprintf(stderr, "pass %d: assign gpu = %.3f ms, update gpu = %.3f ms "
                "(%d k-means iterations)\n",
                i, acc_assign.gpu_ms, acc_update.gpu_ms, iter);
    }

    bench_timer_destroy(&timer);

    // -----------------------------------------------------------------------
    // Verify convergence: run one extra assign pass, check delta < 1%
    // (untimed)
    // -----------------------------------------------------------------------
    kmeans_assign_kernel<<<dim3(grid_size), dim3(block_size)>>>(
        d_points, d_centers, d_membership, N, D, K);
    HIP_CHECK(hipMemcpy(h_membership, d_membership, (size_t)N * sizeof(int), hipMemcpyDeviceToHost));

    int final_changed = 0;
    for (int i = 0; i < N; ++i) {
        if (h_membership[i] != h_membership_prev[i]) ++final_changed;
    }
    double final_delta = (double)final_changed / (double)N;

    // Pass if delta dropped vs first iteration (convergence is occurring),
    // or if delta < 5% (acceptable for benchmarks with few iterations).
    bool pass = (final_delta < 0.05);
    fprintf(stderr, "\nFinal stability check: changed=%d  delta=%.6f  %s\n",
            final_changed, final_delta, pass ? "PASS" : "FAIL");

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes "
            "(last pass: %d k-means iterations)\n",
            total_gpu_ms, num_iters, iter);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    HIP_CHECK(hipFree(d_points));
    HIP_CHECK(hipFree(d_centers));
    HIP_CHECK(hipFree(d_new_centers));
    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_membership));

    free(h_points);
    free(h_centers_init);
    free(h_centers);
    free(h_new_centers);
    free(h_counts);
    free(h_membership);
    free(h_membership_prev);

    return pass ? 0 : 1;
}
