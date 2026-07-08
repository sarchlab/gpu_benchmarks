// parboil_histogram.hip — Parboil Histogram benchmark (HIP, self-contained)
//
// Computes a 256-bin histogram of N random uint32 values using atomic operations.
// Each element is mapped to a bin via: bin = element & 0xFF (lower 8 bits).
//
// Algorithm:
//   - One GPU kernel: each thread reads one element, computes bin, atomicAdd to hist
//   - 256 threads per block, ceil(N/256) blocks
//   - Histogram is zeroed (untimed) before every pass, warmup and timed alike
//   - CPU verify: compare all 256 bins exactly
//
// Default: N = 16*1024*1024 = 16777216 elements
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA).
//
// Usage:
//   ./parboil_histogram [--n N]
//
// Environment variables (override argv):
//   BENCH_PARAM_n, BENCH_PARAM_num_bins, BENCH_PARAM_block_size
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed pass carrying
// the kernel's GPU event time (time_ms) and host wall-clock time
// (host_time_ms), then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. GB/s is derived downstream from time_ms + params.
// Output (stderr): device info, per-pass timing, PASS/FAIL

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// ---------------------------------------------------------------------------
// Inlined utilities
// ---------------------------------------------------------------------------

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
// Constants
// ---------------------------------------------------------------------------

#define NUM_BINS   256
#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// GPU kernel: each thread reads one element and atomically increments the bin
// ---------------------------------------------------------------------------
__global__ void histogram_kernel(const uint32_t* __restrict__ data,
                                  uint32_t* __restrict__ hist,
                                  uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t bin = data[idx] & 0xFF;
    atomicAdd(&hist[bin], 1u);
}

// ---------------------------------------------------------------------------
// CPU reference histogram for verification
// ---------------------------------------------------------------------------
static void cpu_histogram(const uint32_t* data, uint32_t* hist, uint32_t n)
{
    memset(hist, 0, NUM_BINS * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; ++i) {
        hist[data[i] & 0xFF]++;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Default values from params.json
    uint32_t n     = 16 * 1024 * 1024;
    int      num_bins = 256;
    int      block_size = 256;

    // Command-line args as fallback
    n     = (uint32_t)parseIntParam(argc, argv, "--n", (int)n);
    num_bins = parseIntParam(argc, argv, "--num_bins", num_bins);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) n = (uint32_t)atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_bins");
    if (env_val) num_bins = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    size_t bytes_data = (size_t)n * sizeof(uint32_t);
    size_t bytes_hist = (size_t)NUM_BINS * sizeof(uint32_t);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "N = %u elements  |  Bins = %d  |  Iterations: %d warmup + %d timed\n\n",
            n, NUM_BINS, num_warmup, num_iters);

    // Host allocations
    uint32_t* h_data = (uint32_t*)malloc(bytes_data);
    uint32_t* h_hist = (uint32_t*)malloc(bytes_hist);
    uint32_t* h_ref  = (uint32_t*)malloc(bytes_hist);

    // Initialize data with pseudo-random uint32 values
    srand(42);
    for (uint32_t i = 0; i < n; ++i) {
        // Combine multiple rand() calls to fill 32 bits
        h_data[i] = ((uint32_t)rand() ^ ((uint32_t)rand() << 15) ^ ((uint32_t)rand() << 30));
    }

    // Device allocations
    uint32_t *d_data, *d_hist;
    HIP_CHECK(hipMalloc(&d_data, bytes_data));
    HIP_CHECK(hipMalloc(&d_hist, bytes_hist));

    // Copy input data to device (done once, reused across iterations)
    HIP_CHECK(hipMemcpy(d_data, h_data, bytes_data, hipMemcpyHostToDevice));

    uint32_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    char params[128];
    snprintf(params, sizeof(params),
             "\"n\":%u,\"num_bins\":%d,\"block_size\":%d",
             n, num_bins, block_size);

    // Single kernel, but the histogram buffer ACCUMULATES: it must be zeroed
    // before every pass (untimed) or later passes would add onto earlier
    // counts and verification would fail — the old code reset it too. The
    // memset stays outside the timed bracket; only the kernel launch is timed.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t lap;

    auto run_hist_pass = [&]() {
        HIP_CHECK(hipMemset(d_hist, 0, bytes_hist));
        lap = bench_time_op(&timer, [&]() {
            histogram_kernel<<<blocks, BLOCK_SIZE>>>(d_data, d_hist, n);
        });
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_hist_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_hist_pass();
        bench_emit_iter("histogram_kernel", i, &lap, params);
        total_gpu_ms += lap.gpu_ms;
        fprintf(stderr, "histogram_kernel iter %d: gpu = %.3f ms, host = %.3f ms\n",
                i, lap.gpu_ms, lap.host_ms);
    }

    bench_timer_destroy(&timer);

    // Copy result histogram back (from the last pass)
    HIP_CHECK(hipMemcpy(h_hist, d_hist, bytes_hist, hipMemcpyDeviceToHost));

    // CPU reference verification
    cpu_histogram(h_data, h_ref, n);

    bool pass = true;
    for (int b = 0; b < NUM_BINS; ++b) {
        if (h_hist[b] != h_ref[b]) {
            fprintf(stderr, "MISMATCH bin[%d]: gpu=%u cpu=%u\n", b, h_hist[b], h_ref[b]);
            pass = false;
        }
    }
    fprintf(stderr, "Correctness check (all %d bins): %s\n\n",
            NUM_BINS, pass ? "PASS" : "FAIL");

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_hist));
    free(h_data);
    free(h_hist);
    free(h_ref);

    return 0;
}
