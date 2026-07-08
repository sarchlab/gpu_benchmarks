// npb_ep.hip — NAS Parallel Benchmarks Embarrassingly Parallel (EP)
//
// Generates N pairs of Gaussian random deviates using Box-Muller transform
// with a per-thread linear congruential RNG, then counts results into 10
// annular bins.  Purely compute-bound workload.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./npb_ep [--size N]
//
//   --size N         Number of Gaussian pairs (default: 16777216 = 2^24)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_seed
//   BENCH_PARAM_num_warmup   (warmup launches, default 3)
//   BENCH_PARAM_num_iters    (timed launches, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// carrying the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the SUM of all iterations' GPU event times.
// Gpairs/sec is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Number of annular bins
#define NUM_BINS 10

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
// LCG RNG — integer-based, works identically on CPU/GPU/Metal
//   seed = seed * 1103515245 + 12345  (ANSI C LCG, mod 2^32 by overflow)
// ---------------------------------------------------------------------------

__device__ __host__ static inline unsigned int lcg_next(unsigned int seed) {
    return seed * 1103515245u + 12345u;
}

// ---------------------------------------------------------------------------
// Kernel: EP — generate Gaussian pairs and bin them
//   Each thread generates one pair of uniform random numbers, applies
//   Box-Muller transform, and atomically increments the appropriate bin.
//   bins[NUM_BINS] is stored in global memory.
// ---------------------------------------------------------------------------

__global__ void ep_kernel(int N, unsigned long long* bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Per-thread seed derived from thread index
    unsigned int seed = (unsigned int)(idx + 1);
    // Advance RNG a few steps for decorrelation
    seed = lcg_next(seed);
    seed = lcg_next(seed);

    // Generate two uniform random numbers in (0, 1)
    seed = lcg_next(seed);
    float u1 = (float)seed / 4294967296.0f;
    seed = lcg_next(seed);
    float u2 = (float)seed / 4294967296.0f;

    // Avoid log(0)
    if (u1 < 1e-10f) u1 = 1e-10f;

    // Box-Muller transform
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;
    float x1 = r * cosf(theta);
    float x2 = r * sinf(theta);

    // Compute distance squared
    float t = x1 * x1 + x2 * x2;

    // Bin index: floor(sqrt(t)), capped at NUM_BINS-1
    int bin = (int)sqrtf(t);
    if (bin >= NUM_BINS) bin = NUM_BINS - 1;
    if (bin < 0) bin = 0;

    atomicAdd(&bins[bin], 1ULL);
}

// ---------------------------------------------------------------------------
// CPU reference for verification
// ---------------------------------------------------------------------------

static void ep_cpu(int N, unsigned long long* bins) {
    for (int i = 0; i < NUM_BINS; ++i) bins[i] = 0;

    for (int idx = 0; idx < N; ++idx) {
        unsigned int seed = (unsigned int)(idx + 1);
        seed = lcg_next(seed);
        seed = lcg_next(seed);

        seed = lcg_next(seed);
        float u1 = (float)seed / 4294967296.0f;
        seed = lcg_next(seed);
        float u2 = (float)seed / 4294967296.0f;

        if (u1 < 1e-10f) u1 = 1e-10f;

        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * (float)M_PI * u2;
        float x1 = r * cosf(theta);
        float x2 = r * sinf(theta);

        float t = x1 * x1 + x2 * x2;

        int bin = (int)sqrtf(t);
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;
        if (bin < 0) bin = 0;

        bins[bin] += 1;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int N          = 16777216;  // 2^24
    int block_size = 256;
    int seed_param = 271828183;

    // Command-line fallback
    N          = parseIntParam(argc, argv, "--size", N);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_seed");
    if (env_val) seed_param = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "EP pairs: %d (2^%.0f)  |  Iterations: %d warmup + %d timed\n\n",
            N, log2((double)N), num_warmup, num_iters);

    // Device allocation for bins
    unsigned long long* d_bins;
    HIP_CHECK(hipMalloc(&d_bins, NUM_BINS * sizeof(unsigned long long)));

    // Kernel launch config
    int blockSize = block_size;
    int gridSize  = (N + blockSize - 1) / blockSize;

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"seed\":%d",
             N, block_size, seed_param);

    // One EP pass. The kernel derives every thread's RNG seed from its
    // thread index, so each launch performs identical work; the bins are
    // atomically ACCUMULATED, so they are reset (untimed) before each launch
    // to keep the verification below meaningful.
    bench_timer_t timer;
    bench_timer_create(&timer);

    for (int w = 0; w < num_warmup; ++w) {
        HIP_CHECK(hipMemset(d_bins, 0, NUM_BINS * sizeof(unsigned long long)));
        ep_kernel<<<dim3(gridSize), dim3(blockSize)>>>(N, d_bins);
        HIP_CHECK(hipDeviceSynchronize());
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        // Reset bins outside the timed bracket
        HIP_CHECK(hipMemset(d_bins, 0, NUM_BINS * sizeof(unsigned long long)));
        HIP_CHECK(hipDeviceSynchronize());

        bench_lap_t lap = bench_time_op(&timer, [&]() {
            ep_kernel<<<dim3(gridSize), dim3(blockSize)>>>(N, d_bins);
        });

        bench_emit_iter("ep_kernel", i, &lap, params);
        total_gpu_ms += lap.gpu_ms;
        fprintf(stderr, "ep_kernel iter %d: gpu = %.3f ms, host = %.3f ms\n",
                i, lap.gpu_ms, lap.host_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    // -------------------------------------------------------------------
    // Verification: compare GPU bins against CPU reference
    // -------------------------------------------------------------------
    unsigned long long h_bins[NUM_BINS];
    HIP_CHECK(hipMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost));

    // Use a small N for verification to keep CPU time reasonable
    int verifyN = (N <= 1048576) ? N : 1048576;

    // Re-run GPU with verifyN if different
    unsigned long long h_verify_gpu[NUM_BINS];
    if (verifyN != N) {
        int vGridSize = (verifyN + blockSize - 1) / blockSize;
        HIP_CHECK(hipMemset(d_bins, 0, NUM_BINS * sizeof(unsigned long long)));
        ep_kernel<<<dim3(vGridSize), dim3(blockSize)>>>(verifyN, d_bins);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_verify_gpu, d_bins, NUM_BINS * sizeof(unsigned long long),
                            hipMemcpyDeviceToHost));
    } else {
        for (int i = 0; i < NUM_BINS; ++i) h_verify_gpu[i] = h_bins[i];
    }

    unsigned long long cpu_bins[NUM_BINS];
    ep_cpu(verifyN, cpu_bins);

    fprintf(stderr, "\nBin counts (GPU vs CPU, N=%d):\n", verifyN);
    int errors = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        fprintf(stderr, "  Bin %d: GPU=%llu  CPU=%llu\n", i, h_verify_gpu[i], cpu_bins[i]);
        if (h_verify_gpu[i] != cpu_bins[i]) errors++;
    }

    if (errors > 0)
        fprintf(stderr, "FAIL: %d bin mismatches\n", errors);
    else
        fprintf(stderr, "PASS\n");

    HIP_CHECK(hipFree(d_bins));

    return 0;
}
