// shoc_sort.hip — SHOC Sort benchmark (HIP, self-contained)
//
// Bitonic sort on a large array of random unsigned integers.
// O(n log^2 n) compare-and-swap network.
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback:
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_precision
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel","name":"bitonic_step",
// ...} JSON line carrying the pass's accumulated GPU event time (summed over
// all compare-and-swap steps) and host time, then a summary whose
// total_time_ms is the SUM of all recorded GPU event times. Melements/s is
// derived downstream from time_ms and params.
// Output (stderr): Human-readable diagnostics

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Inlined HIP utilities
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

static const char* parseStrParam(int argc, char** argv, const char* name,
                                 const char* defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Bitonic sort kernel
// ---------------------------------------------------------------------------
__global__ void bitonic_step(unsigned int* data, int j, int k) {
    int i = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    int ixj = i ^ j;

    if (ixj > i) {
        bool ascending = ((i & k) == 0);
        unsigned int a = data[i];
        unsigned int b = data[ixj];
        if (ascending ? (a > b) : (a < b)) {
            data[i]   = b;
            data[ixj] = a;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU bitonic sort driver — every compare-and-swap step is bracketed with its
// own event pair and accumulated into `acc`, so one full sort yields one
// accumulated lap for the bitonic_step kernel.
// ---------------------------------------------------------------------------
static void bitonic_sort_gpu(unsigned int* d_data, int N, int block_size,
                             bench_timer_t* timer, bench_lap_t* acc) {
    int grid = (N + block_size - 1) / block_size;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bench_lap_add(acc, bench_time_op(timer, [&]() {
                bitonic_step<<<dim3(grid), dim3(block_size)>>>(d_data, j, k);
            }));
        }
    }
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------
static bool verify_sorted(const unsigned int* arr, int N) {
    for (int i = 1; i < N; ++i) {
        if (arr[i] < arr[i - 1]) {
            fprintf(stderr, "Mismatch at %d: arr[%d]=%u > arr[%d]=%u\n",
                    i, i - 1, arr[i - 1], i, arr[i]);
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

    // Defaults
    int size       = 1 << 22; // 4M
    int block_size = 256;
    const char* precision = "uint";

    // Command-line fallback
    size       = parseIntParam(argc, argv, "--size", size);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);
    precision  = parseStrParam(argc, argv, "--precision", precision);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) precision = env_val;
    // Validate string params so the values echoed into the JSON records are
    // always canonical literals (never raw env text).
    if (strcmp(precision, "uint") != 0 && strcmp(precision, "int") != 0 &&
        strcmp(precision, "float") != 0) {
        fprintf(stderr, "Invalid precision '%s' (use uint, int, float)\n", precision);
        return EXIT_FAILURE;
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int N     = size;

    // Round N up to next power of 2
    {
        int p = 1;
        while (p < N) p <<= 1;
        if (p != N) {
            fprintf(stderr, "Warning: N=%d is not a power of 2, rounding up to %d\n", N, p);
            N = p;
        }
    }

    size_t bytes = (size_t)N * sizeof(unsigned int);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d elements (%.1f MB)\n\n",
            N, (double)bytes / (1024.0 * 1024.0));

    // Host allocation and initialization
    unsigned int* h_data = (unsigned int*)malloc(bytes);
    unsigned int* h_orig = (unsigned int*)malloc(bytes);
    if (!h_data || !h_orig) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_orig[i] = (unsigned int)rand();
    }

    // Device allocation
    unsigned int* d_data;
    HIP_CHECK(hipMalloc(&d_data, bytes));

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"precision\":\"%s\"",
             size, block_size, precision);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;

    // One full sort pass. The sort is IN PLACE, but the bitonic network is
    // DATA-INDEPENDENT: it executes the exact same compare-and-swap sequence
    // regardless of element order, so sorting already-sorted data takes the
    // same time. We still re-upload the pristine unsorted input every pass
    // (untimed, as the old code did) so the post-loop correctness check
    // verifies a genuine sort of random data.
    auto run_sort_pass = [&]() {
        bench_lap_reset(&acc);
        HIP_CHECK(hipMemcpy(d_data, h_orig, bytes, hipMemcpyHostToDevice));
        bitonic_sort_gpu(d_data, N, block_size, &timer, &acc);
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_sort_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_sort_pass();
        bench_emit_iter("bitonic_step", i, &acc, params);
        total_gpu_ms += acc.gpu_ms;
        fprintf(stderr, "pass %d: bitonic_step gpu = %.3f ms, host = %.3f ms\n",
                i, acc.gpu_ms, acc.host_ms);
    }

    bench_timer_destroy(&timer);

    // Correctness check (result of the last pass)
    HIP_CHECK(hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost));
    if (verify_sorted(h_data, N)) {
        fprintf(stderr, "Correctness: PASS\n");
    } else {
        fprintf(stderr, "Correctness: FAIL\n");
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // Cleanup
    HIP_CHECK(hipFree(d_data));
    free(h_data);
    free(h_orig);

    return 0;
}
