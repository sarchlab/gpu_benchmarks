// rodinia_kmeans.hip — Rodinia K-Means Clustering benchmark (HIP, self-contained)
//
// Adapted from Rodinia 3.1 cuda/kmeans at commit
// 366b283456506ef2fe2c2b7dc5e83e941ffc5524 (see README for pinned files).
// invert_mapping retains the upstream row-major -> feature-major mapping;
// kmeansPoint retains the default CPU-reduction kernel's nearest-center
// arithmetic, strict tie order, constant-memory centers and point geometry.
// The bound texture fetch is replaced by __ldg(&features[addr]). This keeps
// the feature address mapping; caching/instruction behavior requires separate
// target qualification. Inactive GPU reduction branches are omitted.
//
// The local measurement driver uses synthetic LCG input and exactly max_iter
// iterations. The pinned driver reads a dataset and uses membership-change
// convergence with a loop guard. Both reduce centroid sums on the CPU in
// ascending point order, retaining old centers for empty clusters. The local
// driver adds the warmer/JSON timing harness and omits unused membership/cluster
// uploads. Texture binding is omitted because feature reads now use __ldg.
//
// Usage:
//   ./rodinia_kmeans [--points N] [--features D] [--clusters K] [--max_iter M]
//
//   --points N     Number of data points   (default: 100000)
//   --features D   Features per point       (default: 34)
//   --clusters K   Number of clusters       (default: 5)
//   --max_iter M   K-means iterations  (default: 10)
//
// Environment variables (override argv):
//   BENCH_PARAM_points, BENCH_PARAM_features, BENCH_PARAM_clusters,
//   BENCH_PARAM_max_iter
//
// Output (stdout): one {"type":"kernel",...} JSON line per
// kernel (invert_mapping, kmeansPoint) carrying that kernel's accumulated GPU
// event time (time_ms) and host time (host_time_ms), then a summary whose
// total_time_ms is the SUM of all recorded GPU event times.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <climits>
#include "cpu_reference.h"

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
// Kernels — from Rodinia kmeans/kmeans_cuda_kernel.cu
// ---------------------------------------------------------------------------

// Cluster centers in constant memory (upstream ASSUMED_NR_CLUSTERS=32,
// max 34 features). Requires nclusters*nfeatures <= 32*34.
#define ASSUMED_NR_CLUSTERS 32
__constant__ float c_clusters[ASSUMED_NR_CLUSTERS * 34];

// invert_mapping: row-major [point][feature] -> column-major [feature][point]
// so consecutive threads read consecutive addresses. The pinned body is retained.
__global__ void invert_mapping(float *input,   /* original */
                               float *output,   /* inverted */
                               int npoints,
                               int nfeatures)
{
    int point_id = threadIdx.x + blockDim.x * blockIdx.x;
    int i;
    if (point_id < npoints) {
        for (i = 0; i < nfeatures; i++)
            output[point_id + npoints * i] = input[point_id * nfeatures + i];
    }
    return;
}

// kmeansPoint: assign each point to its nearest cluster center. The pinned
// default CPU-reduction body is retained except for the texture -> __ldg
// substitution. Inactive GPU_DELTA_REDUCTION / GPU_NEW_CENTER_REDUCTION
// branches are omitted (see README).
__global__ void
kmeansPoint(float  *features,           /* in: inverted [nfeatures*npoints] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
            float  *clusters,
            float  *block_clusters,
            int    *block_deltas)
{
    // block ID
    const unsigned int block_id = gridDim.x * blockIdx.y + blockIdx.x;
    // point/thread ID
    const unsigned int point_id = block_id * blockDim.x * blockDim.y + threadIdx.x;

    int index = -1;

    if (point_id < npoints)
    {
        int i, j;
        float min_dist = FLT_MAX;
        float dist;                                             /* distance square between a point to cluster center */

        /* find the cluster center id with min distance to pt */
        for (i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures;             /* base index of cluster centers for inverted array */
            float ans = 0.0;                                    /* Euclidean distance square */

            for (j = 0; j < nfeatures; j++)
            {
                int addr = point_id + j * npoints;              /* appropriate index of data point (inverted layout) */
                /* tex1Dfetch(t_features, addr) replaced by __ldg (see header) */
                float diff = (__ldg(&features[addr]) -
                              c_clusters[cluster_base_index + j]);  /* distance between a data point to cluster centers */
                ans += diff * diff;                             /* sum of squares */
            }
            dist = ans;

            /* see if distance is smaller than previous ones:
               if so, change minimum distance and save index of cluster center */
            if (dist < min_dist) {
                min_dist = dist;
                index    = i;
            }
        }
    }

    if (point_id < npoints)
    {
        /* assign the membership to object point_id */
        membership[point_id] = index;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int npoints   = parseIntParam(argc, argv, "--points",   100000);
    int nfeatures = parseIntParam(argc, argv, "--features",     34);
    int nclusters = parseIntParam(argc, argv, "--clusters",      5);
    int max_iter  = parseIntParam(argc, argv, "--max_iter",     10);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_points");   if (env_val) npoints   = atoi(env_val);
    env_val = getenv("BENCH_PARAM_features"); if (env_val) nfeatures = atoi(env_val);
    env_val = getenv("BENCH_PARAM_clusters"); if (env_val) nclusters = atoi(env_val);
    env_val = getenv("BENCH_PARAM_max_iter"); if (env_val) max_iter  = atoi(env_val);

    if (npoints < 1 || nfeatures < 1 || nclusters < 1 || max_iter < 1) {
        fprintf(stderr, "Invalid k-means parameters: points=%d features=%d clusters=%d max_iter=%d; all must be positive\n",
                npoints, nfeatures, nclusters, max_iter);
        return EXIT_FAILURE;
    }
    const long long center_elements = static_cast<long long>(nclusters) * nfeatures;
    if (center_elements > ASSUMED_NR_CLUSTERS * 34) {
        fprintf(stderr, "clusters*features (%lld) exceeds constant-memory capacity (%d)\n",
                center_elements, ASSUMED_NR_CLUSTERS * 34);
        return EXIT_FAILURE;
    }
    // Initial centers are copied from the first nclusters complete input points.
    if (nclusters > npoints) {
        fprintf(stderr, "Invalid k-means parameters: clusters=%d exceeds points=%d\n",
                nclusters, npoints);
        return EXIT_FAILURE;
    }
    const long long feature_elements = static_cast<long long>(npoints) * nfeatures;
    if (feature_elements > INT_MAX) {
        fprintf(stderr, "Invalid k-means parameters: points*features=%lld exceeds signed-int indexing\n",
                feature_elements);
        return EXIT_FAILURE;
    }

    // Grid geometry retained from pinned allocateMemory/kmeansCuda.
    const int num_threads_perdim = 16;                          // THREADS_PER_DIM
    const int num_threads = num_threads_perdim * num_threads_perdim;  // 256
    int num_blocks = npoints / num_threads;
    if (npoints % num_threads > 0) num_blocks++;
    int num_blocks_perdim = (int)sqrt((double)num_blocks);
    while (num_blocks_perdim * num_blocks_perdim < num_blocks) num_blocks_perdim++;
    num_blocks = num_blocks_perdim * num_blocks_perdim;

    // invert_mapping stores even padded threads' point IDs in signed int.
    const long long padded_points = static_cast<long long>(num_blocks) * num_threads;
    if (padded_points - 1 > INT_MAX) {
        fprintf(stderr, "Invalid k-means parameters: padded point count=%lld exceeds signed-int indexing\n",
                padded_points);
        return EXIT_FAILURE;
    }

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "points=%d  features=%d  clusters=%d  max_iter=%d\n\n",
            npoints, nfeatures, nclusters, max_iter);

    // Host data (row-major [point][feature]).
    size_t feat_bytes    = (size_t)npoints * nfeatures * sizeof(float);
    size_t centers_bytes = (size_t)nclusters * nfeatures * sizeof(float);
    float* h_features    = (float*)malloc(feat_bytes);
    float* h_centers     = (float*)malloc(centers_bytes);
    float* h_centers_ini = (float*)malloc(centers_bytes);
    size_t membership_elements = 0;
    try { membership_elements = kmeans_reference::history_elements(npoints, max_iter); }
    catch (const std::exception& error) {
        fprintf(stderr, "K-means allocation failed: %s\n", error.what());
        free(h_features); free(h_centers); free(h_centers_ini);
        return EXIT_FAILURE;
    }
    int* h_membership_all = (int*)malloc(membership_elements * sizeof(int));
    if (!h_features || !h_centers || !h_centers_ini || !h_membership_all) {
        fprintf(stderr, "K-means host allocation failed for points=%d features=%d clusters=%d max_iter=%d\n",
                npoints, nfeatures, nclusters, max_iter);
        free(h_features); free(h_centers); free(h_centers_ini); free(h_membership_all);
        return EXIT_FAILURE;
    }

    unsigned int seed = 42;
    for (int i = 0; i < npoints * nfeatures; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h_features[i] = (float)(seed >> 8) / (float)(1 << 24);   // [0,1)
    }
    // Initial cluster centers = first nclusters points (kept pristine).
    memcpy(h_centers_ini, h_features, centers_bytes);

    // Device buffers.
    float* feature_flipped_d;   // original (row-major)
    float* feature_d;           // inverted (feature-major)
    float* clusters_d;
    int*   membership_d;
    HIP_CHECK(hipMalloc((void**)&feature_flipped_d, feat_bytes));
    HIP_CHECK(hipMalloc((void**)&feature_d,         feat_bytes));
    HIP_CHECK(hipMalloc((void**)&clusters_d,        centers_bytes));
    HIP_CHECK(hipMalloc((void**)&membership_d,      (size_t)npoints * sizeof(int)));
    HIP_CHECK(hipMemcpy(feature_flipped_d, h_features, feat_bytes, hipMemcpyHostToDevice));

    dim3 grid(num_blocks_perdim, num_blocks_perdim);
    dim3 threads(num_threads);

    char params[256];
    snprintf(params, sizeof(params),
             "\"points\":%d,\"features\":%d,\"clusters\":%d,\"max_iter\":%d,\"threads_per_block\":%d,\"verification\":\"full_cpu_fma_trajectory_v1\"",
             npoints, nfeatures, nclusters, max_iter, num_threads);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_invert, acc_assign;

    // One recorded pass = a full clustering run: invert the features once, then
    // max_iter assignment iterations. Cluster centers are RE-INIT to the
    // pristine initial centers at pass start (untimed) so every pass converges
    // over identical work. Per iteration: upload centers to constant memory
    // (untimed), launch kmeansPoint (timed), copy membership back and recompute
    // centers as per-cluster means on the CPU (untimed). invert_mapping reads
    // immutable features; it is re-run each pass only so it emits a timing
    // record — the result is identical every pass.
    auto run_pass = [&]() -> bool {
        bench_lap_reset(&acc_invert);
        bench_lap_reset(&acc_assign);
        memcpy(h_centers, h_centers_ini, centers_bytes);

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_invert, bench_time_op(&timer, [&]() {
            invert_mapping<<<num_blocks, num_threads>>>(feature_flipped_d, feature_d,
                                                        npoints, nfeatures);
        }));

        for (int iter = 0; iter < max_iter; ++iter) {
            // Upload current centers to constant memory (untimed).
            HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_clusters), h_centers,
                                        centers_bytes, 0, hipMemcpyHostToDevice));

            bench_lap_add(&acc_assign, bench_time_op(&timer, [&]() {
                kmeansPoint<<<grid, threads>>>(feature_d, nfeatures, npoints, nclusters,
                                               membership_d, clusters_d, nullptr, nullptr);
            }));

            // Retain each measured assignment without adding an in-loop copy.
            int* h_membership = h_membership_all + static_cast<size_t>(iter) * npoints;
            // Copy membership back and recompute centroids on the CPU (untimed).
            HIP_CHECK(hipMemcpy(h_membership, membership_d,
                                (size_t)npoints * sizeof(int), hipMemcpyDeviceToHost));

            float* new_centers = (float*)calloc((size_t)nclusters * nfeatures, sizeof(float));
            int*   counts      = (int*)calloc((size_t)nclusters, sizeof(int));
            if (!new_centers || !counts) {
                fprintf(stderr, "K-means centroid allocation failed at iteration %d\n", iter);
                free(new_centers); free(counts);
                return false;
            }
            for (int p = 0; p < npoints; ++p) {
                int c = h_membership[p];
                if (c < 0 || c >= nclusters) {
                    fprintf(stderr, "K-means verification: FAIL invalid membership[%d][%d]=%d\n", iter, p, c);
                    free(new_centers); free(counts);
                    return false;
                }
                counts[c]++;
                for (int f = 0; f < nfeatures; ++f)
                    new_centers[c * nfeatures + f] += h_features[p * nfeatures + f];
            }
            for (int c = 0; c < nclusters; ++c) {
                if (counts[c] > 0) {
                    for (int f = 0; f < nfeatures; ++f)
                        h_centers[c * nfeatures + f] = new_centers[c * nfeatures + f] / counts[c];
                }
                // empty cluster: keep previous center
            }
            free(new_centers);
            free(counts);
        }
        HIP_CHECK(hipGetLastError());
        return true;
    };

    bool verified = run_pass();
    // All verification is after measured work. References consume only pristine
    // row-major features and initial centers; no GPU output feeds the oracle.
    if (verified) {
        size_t values = 0, mismatches = 0;
        auto check_float = [&](const char* field, size_t at, float actual, float expected) {
            ++values;
            if (!kmeans_reference::value_matches(actual, expected)) {
                if (mismatches < 10)
                    fprintf(stderr, "MISMATCH %s[%zu]: actual=%.9g expected=%.9g\n", field, at, actual, expected);
                ++mismatches;
            }
        };
        try {
            {
                std::vector<float> column(npoints);
                for (int f = 0; f < nfeatures; ++f) {
                    HIP_CHECK(hipMemcpy(column.data(), feature_d + static_cast<size_t>(f) * npoints,
                                        static_cast<size_t>(npoints) * sizeof(float), hipMemcpyDeviceToHost));
                    for (int p = 0; p < npoints; ++p)
                        check_float("inversion", static_cast<size_t>(f) * npoints + p,
                                    column[p], h_features[static_cast<size_t>(p) * nfeatures + f]);
                }
            }
            const auto reference = kmeans_reference::trajectory(h_features, h_centers_ini,
                                                                 npoints, nfeatures, nclusters, max_iter);
            for (size_t at = 0; at < membership_elements; ++at) {
                ++values;
                if (h_membership_all[at] != reference.membership[at]) {
                    if (mismatches < 10)
                        fprintf(stderr, "MISMATCH membership[%zu][%zu]: actual=%d expected=%d\n",
                                at / npoints, at % npoints, h_membership_all[at], reference.membership[at]);
                    ++mismatches;
                }
            }
            for (size_t at = 0; at < static_cast<size_t>(nclusters) * nfeatures; ++at)
                check_float("centers", at, h_centers[at], reference.centers[at]);
            verified = mismatches == 0;
            fprintf(stderr, "K-means verification: %s (%zu values, %zu mismatches)\n",
                    verified ? "PASS" : "FAIL", values, mismatches);
        } catch (const std::exception& error) {
            fprintf(stderr, "K-means verification: FAIL reference unavailable: %s\n", error.what());
            verified = false;
        }
    }

    double total_gpu_ms = 0.0;
    bench_emit_iter("invert_mapping", 0, &acc_invert, params);
    bench_emit_iter("kmeansPoint",    0, &acc_assign, params);
    total_gpu_ms += acc_invert.gpu_ms + acc_assign.gpu_ms;
    fprintf(stderr, "invert_mapping gpu = %.3f ms, kmeansPoint gpu = %.3f ms\n",
            acc_invert.gpu_ms, acc_assign.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over 1 pass (%d k-means iters/pass)\n",
            total_gpu_ms, max_iter);

    HIP_CHECK(hipFree(feature_flipped_d));
    HIP_CHECK(hipFree(feature_d));
    HIP_CHECK(hipFree(clusters_d));
    HIP_CHECK(hipFree(membership_d));
    free(h_features);
    free(h_centers);
    free(h_centers_ini);
    free(h_membership_all);

    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
