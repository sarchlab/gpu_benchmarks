// npb_is.hip — NAS Parallel Benchmarks Integer Sort (IS)
//
// Parallel bucket sort of N random integers in range [0, MAX_KEY).
// Phase 1: histogram (count per bucket)
// Phase 2: prefix sum (exclusive scan) on histogram
// Phase 3: scatter keys to sorted positions
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./npb_is [--size N]
//
//   --size N         Number of keys (default: 8388608 = 2^23)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_verify
//   BENCH_PARAM_num_warmup   (warmup sort passes, default 3)
//   BENCH_PARAM_num_iters    (timed sort passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (histogram_kernel, prefix_sum_kernel, scatter_kernel) carrying that
// kernel's GPU event time (time_ms) and host wall-clock time (host_time_ms)
// within the pass, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. Keys/sec is derived downstream from time_ms and
// params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define NUM_BUCKETS 1024    // 2^10
#define MAX_KEY     524288  // 2^19

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

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
// Kernel: histogram — count keys per bucket
//   Each key maps to bucket = key / (MAX_KEY / NUM_BUCKETS)
// ---------------------------------------------------------------------------

__global__ void histogram_kernel(const int* __restrict__ keys, int N,
                                 unsigned int* __restrict__ hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int bucket = keys[idx] / (MAX_KEY / NUM_BUCKETS);
    if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;
    atomicAdd(&hist[bucket], 1u);
}

// ---------------------------------------------------------------------------
// Kernel: prefix sum (exclusive scan) — simple single-block scan
//   Works for NUM_BUCKETS <= 1024 (one block)
// ---------------------------------------------------------------------------

__global__ void prefix_sum_kernel(unsigned int* hist, int n) {
    extern __shared__ unsigned int sdata[];

    int tid = threadIdx.x;
    if (tid < n)
        sdata[tid] = hist[tid];
    else
        sdata[tid] = 0;
    __syncthreads();

    // Up-sweep (reduce)
    for (int stride = 1; stride < n; stride <<= 1) {
        int index = (tid + 1) * (stride << 1) - 1;
        if (index < n) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    // Set last element to 0 for exclusive scan
    if (tid == 0) sdata[n - 1] = 0;
    __syncthreads();

    // Down-sweep
    for (int stride = n >> 1; stride >= 1; stride >>= 1) {
        int index = (tid + 1) * (stride << 1) - 1;
        if (index < n) {
            unsigned int temp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }

    if (tid < n)
        hist[tid] = sdata[tid];
}

// ---------------------------------------------------------------------------
// Kernel: scatter keys to sorted positions
//   Uses atomicAdd on prefix-sum offsets to get unique output positions
// ---------------------------------------------------------------------------

__global__ void scatter_kernel(const int* __restrict__ keys, int N,
                               unsigned int* __restrict__ offsets,
                               int* __restrict__ sorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int bucket = keys[idx] / (MAX_KEY / NUM_BUCKETS);
    if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;

    unsigned int pos = atomicAdd(&offsets[bucket], 1u);
    sorted[pos] = keys[idx];
}

// ---------------------------------------------------------------------------
// CPU reference: generate keys with simple LCG
// ---------------------------------------------------------------------------

static void generate_keys(int* keys, int N) {
    unsigned int seed = 314159265u;
    for (int i = 0; i < N; ++i) {
        seed = seed * 1103515245u + 12345u;
        keys[i] = (int)((seed >> 4) % MAX_KEY);
    }
}

// ---------------------------------------------------------------------------
// CPU verification: check sorted output is non-decreasing within buckets
// ---------------------------------------------------------------------------

static bool verify_sorted(const int* sorted, int N) {
    for (int i = 1; i < N; ++i) {
        int bucket_prev = sorted[i - 1] / (MAX_KEY / NUM_BUCKETS);
        int bucket_curr = sorted[i] / (MAX_KEY / NUM_BUCKETS);
        if (bucket_curr < bucket_prev) {
            fprintf(stderr, "Verification failed at index %d: %d (bucket %d) > %d (bucket %d)\n",
                    i, sorted[i - 1], bucket_prev, sorted[i], bucket_curr);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int N          = 8388608;   // 2^23
    int block_size = 256;
    const char* verify = "true";

    // Command-line fallback
    N          = parseIntParam(argc, argv, "--size", N);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_verify");
    if (env_val) verify = env_val;
    // Validate string params against their known sets so the values echoed
    // into the JSON records are always canonical literals.
    if (strcmp(verify, "true") != 0 && strcmp(verify, "false") != 0 &&
        strcmp(verify, "skip") != 0) {
        fprintf(stderr, "Invalid verify '%s' (use true, false, skip)\n", verify);
        return 1;
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "IS keys: %d (2^%.0f)  |  Buckets: %d  |  "
            "Iterations: %d warmup + %d timed\n\n",
            N, log2((double)N), NUM_BUCKETS, num_warmup, num_iters);

    size_t keysBytes = (size_t)N * sizeof(int);
    size_t histBytes = NUM_BUCKETS * sizeof(unsigned int);

    // Host allocation + key generation
    int* h_keys   = (int*)malloc(keysBytes);
    int* h_sorted = (int*)malloc(keysBytes);
    generate_keys(h_keys, N);

    // Device allocation
    int* d_keys;
    int* d_sorted;
    unsigned int* d_hist;
    unsigned int* d_offsets;  // copy of prefix-sum for scatter

    HIP_CHECK(hipMalloc(&d_keys, keysBytes));
    HIP_CHECK(hipMalloc(&d_sorted, keysBytes));
    HIP_CHECK(hipMalloc(&d_hist, histBytes));
    HIP_CHECK(hipMalloc(&d_offsets, histBytes));

    // Copy keys to device
    HIP_CHECK(hipMemcpy(d_keys, h_keys, keysBytes, hipMemcpyHostToDevice));

    // Kernel launch config
    int blockSize = block_size;
    int gridKeys  = (N + blockSize - 1) / blockSize;

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"verify\":\"%s\"",
             N, block_size, verify);

    // Per-kernel accumulators: one full IS pass launches each kernel once;
    // each pass's per-kernel GPU/host time becomes that pass's record.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_hist, acc_prefix, acc_scatter;

    // One full IS pass (histogram + prefix sum + scatter). The input keys
    // are never mutated (the sort scatters into d_sorted), the histogram is
    // zeroed and the offsets recopied every pass, so each pass performs
    // identical work. Resets and the d2d copy stay outside timed brackets.
    auto run_is_pass = [&]() {
        bench_lap_reset(&acc_hist);
        bench_lap_reset(&acc_prefix);
        bench_lap_reset(&acc_scatter);

        // Clear histogram (untimed)
        HIP_CHECK(hipMemset(d_hist, 0, histBytes));

        // Phase 1: histogram
        bench_lap_add(&acc_hist, bench_time_op(&timer, [&]() {
            histogram_kernel<<<dim3(gridKeys), dim3(blockSize)>>>(d_keys, N, d_hist);
        }));

        // Phase 2: prefix sum (exclusive scan on histogram)
        // Single-block scan, NUM_BUCKETS threads
        bench_lap_add(&acc_prefix, bench_time_op(&timer, [&]() {
            prefix_sum_kernel<<<dim3(1), dim3(NUM_BUCKETS), NUM_BUCKETS * sizeof(unsigned int), 0>>>(
                d_hist, NUM_BUCKETS);
        }));

        // Copy prefix sums to offsets (scatter will modify offsets; untimed)
        HIP_CHECK(hipMemcpy(d_offsets, d_hist, histBytes, hipMemcpyDeviceToDevice));

        // Phase 3: scatter
        bench_lap_add(&acc_scatter, bench_time_op(&timer, [&]() {
            scatter_kernel<<<dim3(gridKeys), dim3(blockSize)>>>(d_keys, N, d_offsets, d_sorted);
        }));
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_is_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_is_pass();
        bench_emit_iter("histogram_kernel",  i, &acc_hist,    params);
        bench_emit_iter("prefix_sum_kernel", i, &acc_prefix,  params);
        bench_emit_iter("scatter_kernel",    i, &acc_scatter, params);
        total_gpu_ms += acc_hist.gpu_ms + acc_prefix.gpu_ms + acc_scatter.gpu_ms;
        fprintf(stderr, "pass %d: histogram gpu = %.3f ms, prefix_sum gpu = %.3f ms, "
                "scatter gpu = %.3f ms\n",
                i, acc_hist.gpu_ms, acc_prefix.gpu_ms, acc_scatter.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // -------------------------------------------------------------------
    // Verification: check sorted output is bucket-ordered
    // -------------------------------------------------------------------
    HIP_CHECK(hipMemcpy(h_sorted, d_sorted, keysBytes, hipMemcpyDeviceToHost));

    if (verify_sorted(h_sorted, N))
        fprintf(stderr, "PASS\n");
    else
        fprintf(stderr, "FAIL\n");

    HIP_CHECK(hipFree(d_keys));
    HIP_CHECK(hipFree(d_sorted));
    HIP_CHECK(hipFree(d_hist));
    HIP_CHECK(hipFree(d_offsets));
    free(h_keys);
    free(h_sorted);

    return 0;
}
