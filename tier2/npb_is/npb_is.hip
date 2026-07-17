// npb_is.hip — NAS Parallel Benchmarks Integer Sort (IS), faithful port
//
// Faithful GPU port of the NPB IS benchmark (reference: NPB 2.3/3.0 serial
// C `is.c`, non-bucket version, M. Yarrow / S. Satoh):
//   - Keys generated with NPB's 46-bit randlc LCG (seed 314159265,
//     multiplier 1220703125); each key is the sum of four uniform draws
//     scaled by MAX_KEY/4, giving the benchmark's near-gaussian key model.
//   - The timed work is MAX_ITERATIONS(=10) calls of rank(iteration). Each
//     rank() perturbs two keys (key_array[i]=i, key_array[i+10]=MAX_KEY-i),
//     clears the work array, histograms all keys over the full [0,MAX_KEY)
//     range (copying key_array into key_buff2 in the same pass, as upstream
//     does), and computes key ranks with an inclusive prefix sum over
//     MAX_KEY entries. Partial verification checks five test-key ranks per
//     iteration.
//   - One untimed rank(1) precedes the timed loop (upstream's free
//     iteration), and full_verify (the actual permutation + sortedness
//     check) runs untimed after it, exactly as upstream.
//
// Problem classes: size (=N, number of keys) is the scaling parameter. The
// NPB class sizes S/W/A/B/C (2^16/2^20/2^23/2^25/2^27 keys) use the class's
// exact MAX_KEY and published partial-verification tables. Other sizes use
// MAX_KEY = N/16 (the W/A/B/C ratio) and are verified against a bit-exact
// host mirror of the upstream serial algorithm instead of published tables.
// The default N = 8388608 = 2^23 is NPB Class A.
//
// GPU mapping (the only departures from the serial reference, all permitted
// re-parallelizations of the same algorithm):
//   - The MAX_KEY-wide inclusive scan is decomposed into three kernels
//     (per-block scan, block-sum scan, offset add).
//   - The per-iteration key perturbation is a 1-thread kernel.
//   - Partial-verify comparisons run on the host (five 4-byte D2H reads per
//     iteration, untimed); upstream's USE_BUCKETS cache optimization is not
//     used (matching the reference's default).
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./npb_is [--size N]
//
//   --size N         Number of keys (default: 8388608 = 2^23 = Class A)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size (drives launch config),
//   BENCH_PARAM_verify (debug knob)
//
// Output (stdout): one {"type":"kernel",...} JSON line per kernel
// (modify_keys_kernel, clear_kernel, histogram_kernel, scan_blocks_kernel,
// scan_sums_kernel, scan_add_kernel), each carrying that kernel's GPU event
// time (time_ms) and host time (host_time_ms) accumulated across the 10
// timed rank iterations, then a summary whose total_time_ms is the SUM of
// all recorded GPU event times. Keys-ranked/sec is derived downstream.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Constants (upstream is.c)
// ---------------------------------------------------------------------------

#define MAX_ITERATIONS  10
#define TEST_ARRAY_SIZE 5

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// NPB problem classes: published sizes and partial-verification tables
// (upstream is.c, classes S/W/A/B/C)
// ---------------------------------------------------------------------------

typedef struct {
    char cls;
    int  total_keys;
    int  max_key;
    int  test_index[TEST_ARRAY_SIZE];
    int  test_rank[TEST_ARRAY_SIZE];
} npb_class_t;

static const npb_class_t NPB_CLASSES[] = {
    {'S', 1 << 16, 1 << 11,
     {48427, 17148, 23627, 62548, 4431},
     {0, 18, 346, 64917, 65463}},
    {'W', 1 << 20, 1 << 16,
     {357773, 934767, 875723, 898999, 404505},
     {1249, 11698, 1039987, 1043896, 1048018}},
    {'A', 1 << 23, 1 << 19,
     {2112377, 662041, 5336171, 3642833, 4250760},
     {104, 17523, 123928, 8288932, 8388264}},
    {'B', 1 << 25, 1 << 21,
     {41869, 812306, 5102857, 18232239, 26860214},
     {33422937, 10244, 59149, 33135281, 99}},
    {'C', 1 << 27, 1 << 23,
     {44172927, 72999161, 74326391, 129606274, 21736814},
     {61147, 882988, 266290, 133997595, 133525895}},
};

// Published partial-verify expectation for test key i at `iteration`
// (upstream rank(), per-class shift rules).
static int published_expected(char cls, int i, int iteration, int base) {
    switch (cls) {
        case 'S': return (i <= 2) ? base + iteration : base - iteration;
        case 'W': return (i < 2) ? base + (iteration - 2) : base - iteration;
        case 'A': return (i <= 2) ? base + (iteration - 1)
                                  : base - (iteration - 1);
        case 'B': return (i == 1 || i == 2 || i == 4) ? base + iteration
                                                      : base - iteration;
        case 'C': return (i <= 2) ? base + iteration : base - iteration;
    }
    return base;
}

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
// randlc — NPB's portable 46-bit linear congruential generator
// (verbatim upstream semantics)
// ---------------------------------------------------------------------------

static double randlc(double* X, double A) {
    static const double R23 = 1.0 / 8388608.0;            // 2^-23
    static const double R46 = R23 * R23;                  // 2^-46
    static const double T23 = 8388608.0;                  // 2^23
    static const double T46 = T23 * T23;                  // 2^46
    double T1, T2, T3, T4, A1, A2, X1, X2, Z;
    int j;

    // Break A into two parts such that A = 2^23 * A1 + A2
    T1 = R23 * A;
    j = (int)T1;
    A1 = j;
    A2 = A - T23 * A1;

    // Break X into two parts, compute Z = A1*X2 + A2*X1 (mod 2^23),
    // then X = 2^23*Z + A2*X2 (mod 2^46)
    T1 = R23 * *X;
    j = (int)T1;
    X1 = j;
    X2 = *X - T23 * X1;
    T1 = A1 * X2 + A2 * X1;

    j = (int)(R23 * T1);
    T2 = j;
    Z = T1 - T23 * T2;
    T3 = T23 * Z + A2 * X2;
    j = (int)(R46 * T3);
    T4 = j;
    *X = T3 - T46 * T4;
    return R46 * *X;
}

// create_seq — each key is MAX_KEY/4 times the sum of four uniform draws
// (upstream create_seq; seed 314159265, multiplier 1220703125)
static void create_seq(int* key_array, int N, int max_key,
                       double seed, double a) {
    int k = max_key / 4;
    for (int i = 0; i < N; ++i) {
        double x = randlc(&seed, a);
        x += randlc(&seed, a);
        x += randlc(&seed, a);
        x += randlc(&seed, a);
        key_array[i] = (int)(k * x);
    }
}

// ---------------------------------------------------------------------------
// Kernel: per-iteration key perturbation (upstream rank() prologue)
// ---------------------------------------------------------------------------

__global__ void modify_keys_kernel(int* key_array, int iteration, int max_key) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        key_array[iteration] = iteration;
        key_array[iteration + MAX_ITERATIONS] = max_key - iteration;
    }
}

// ---------------------------------------------------------------------------
// Kernel: clear the work array (upstream's key_buff1 zeroing, timed as part
// of rank(), done as a kernel rather than a memset per repo policy)
// ---------------------------------------------------------------------------

__global__ void clear_kernel(int* key_buff1, int max_key) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_key; i += stride)
        key_buff1[i] = 0;
}

// ---------------------------------------------------------------------------
// Kernel: histogram over the full key range; also copies key_array into
// key_buff2 in the same pass, exactly like upstream's fused loop
// ---------------------------------------------------------------------------

__global__ void histogram_kernel(const int* __restrict__ key_array,
                                 int* __restrict__ key_buff2,
                                 int* __restrict__ key_buff1, int N) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        int k = key_array[i];
        key_buff2[i] = k;
        atomicAdd(&key_buff1[k], 1);
    }
}

// ---------------------------------------------------------------------------
// Inclusive prefix sum over MAX_KEY entries (upstream's rank computation
// key_buff1[i+1] += key_buff1[i]), parallelized as a three-kernel
// block-scan / sum-scan / offset-add decomposition. Blelloch work-efficient
// scan per tile of 2*blockDim elements.
// ---------------------------------------------------------------------------

// Exclusive Blelloch scan of sdata[0..tile) in shared memory.
// Returns the tile total (valid in every thread). tile must be a power of 2.
__device__ int block_exclusive_scan(int* sdata, int tile) {
    int t = threadIdx.x;
    int offset = 1;
    for (int d = tile >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1;
            int bi = offset * (2 * t + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }
    __syncthreads();
    int total = sdata[tile - 1];
    __syncthreads();
    if (t == 0) sdata[tile - 1] = 0;
    for (int d = 1; d < tile; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1;
            int bi = offset * (2 * t + 2) - 1;
            int tmp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += tmp;
        }
    }
    __syncthreads();
    return total;
}

// Per-block inclusive scan over tiles of 2*blockDim elements; block totals
// go to block_sums.
__global__ void scan_blocks_kernel(int* data, int* block_sums, int n) {
    extern __shared__ int sdata[];
    const int tile = blockDim.x * 2;
    const int base = blockIdx.x * tile;
    const int t = threadIdx.x;

    int a0 = (base + 2 * t < n) ? data[base + 2 * t] : 0;
    int a1 = (base + 2 * t + 1 < n) ? data[base + 2 * t + 1] : 0;
    sdata[2 * t] = a0;
    sdata[2 * t + 1] = a1;

    int total = block_exclusive_scan(sdata, tile);
    if (t == 0) block_sums[blockIdx.x] = total;

    // exclusive -> inclusive: add back each element's own value
    if (base + 2 * t < n) data[base + 2 * t] = sdata[2 * t] + a0;
    if (base + 2 * t + 1 < n) data[base + 2 * t + 1] = sdata[2 * t + 1] + a1;
}

// Single-block exclusive scan of block_sums[0..nsums), chunked with a
// running carry so nsums may exceed one tile.
__global__ void scan_sums_kernel(int* sums, int nsums) {
    extern __shared__ int sdata[];
    __shared__ int carry;
    const int tile = blockDim.x * 2;
    const int t = threadIdx.x;

    if (t == 0) carry = 0;
    __syncthreads();

    for (int base = 0; base < nsums; base += tile) {
        int a0 = (base + 2 * t < nsums) ? sums[base + 2 * t] : 0;
        int a1 = (base + 2 * t + 1 < nsums) ? sums[base + 2 * t + 1] : 0;
        sdata[2 * t] = a0;
        sdata[2 * t + 1] = a1;

        int total = block_exclusive_scan(sdata, tile);

        if (base + 2 * t < nsums) sums[base + 2 * t] = sdata[2 * t] + carry;
        if (base + 2 * t + 1 < nsums)
            sums[base + 2 * t + 1] = sdata[2 * t + 1] + carry;
        __syncthreads();
        if (t == 0) carry += total;
        __syncthreads();
    }
}

// Add each block's scanned offset to its tile.
__global__ void scan_add_kernel(int* data, const int* block_sums, int n) {
    const int tile = blockDim.x * 2;
    const int base = blockIdx.x * tile;
    const int t = threadIdx.x;
    int off = block_sums[blockIdx.x];
    if (base + 2 * t < n) data[base + 2 * t] += off;
    if (base + 2 * t + 1 < n) data[base + 2 * t + 1] += off;
}

// ---------------------------------------------------------------------------
// Host mirror of upstream serial rank(): perturb keys, capture the test
// values, histogram, inclusive scan. Used as the verification reference for
// every size (and cross-checked against the published tables at class sizes).
// ---------------------------------------------------------------------------

static void ref_rank(int* h_key, int* ref_buff, int N, int max_key,
                     int iteration, const int* test_index, int* pv_vals) {
    h_key[iteration] = iteration;
    h_key[iteration + MAX_ITERATIONS] = max_key - iteration;
    for (int i = 0; i < TEST_ARRAY_SIZE; ++i)
        pv_vals[i] = h_key[test_index[i]];
    memset(ref_buff, 0, (size_t)max_key * sizeof(int));
    for (int i = 0; i < N; ++i)
        ref_buff[h_key[i]]++;
    for (int i = 0; i < max_key - 1; ++i)
        ref_buff[i + 1] += ref_buff[i];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json (Class A)
    int N          = 8388608;   // 2^23
    int block_size = 256;

    // Command-line fallback
    N = parseIntParam(argc, argv, "--size", N);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    // Debug knob (not a sweep parameter): verification runs by default;
    // "false"/"0"/"skip" skip it.
    bool do_verify = true;
    env_val = getenv("BENCH_PARAM_verify");
    if (env_val && (strcmp(env_val, "false") == 0 ||
                    strcmp(env_val, "0")     == 0 ||
                    strcmp(env_val, "skip")  == 0)) {
        do_verify = false;
    }

    if (N < 64) {
        fprintf(stderr, "size must be >= 64 (got %d)\n", N);
        printf("{\"type\":\"error\",\"message\":\"size must be >= 64\"}\n");
        return EXIT_FAILURE;
    }
    if (block_size < 32 || (block_size & (block_size - 1)) != 0) {
        fprintf(stderr, "block_size must be a power of two >= 32 (got %d)\n",
                block_size);
        printf("{\"type\":\"error\",\"message\":\"invalid block_size\"}\n");
        return EXIT_FAILURE;
    }

    // Resolve the NPB class (exact published config) or derive MAX_KEY from
    // the W/A/B/C size ratio (MAX_KEY = N/16) for non-class sizes.
    const npb_class_t* klass = NULL;
    for (size_t c = 0; c < sizeof(NPB_CLASSES) / sizeof(NPB_CLASSES[0]); ++c)
        if (NPB_CLASSES[c].total_keys == N) klass = &NPB_CLASSES[c];

    int max_key;
    int test_index[TEST_ARRAY_SIZE];
    if (klass) {
        max_key = klass->max_key;
        memcpy(test_index, klass->test_index, sizeof(test_index));
    } else {
        max_key = N / 16;
        // Deterministic in-range test positions for non-class sizes (spot
        // checks only; expected values come from the host mirror).
        for (int i = 0; i < TEST_ARRAY_SIZE; ++i)
            test_index[i] = NPB_CLASSES[2].test_index[i] % N;  // Class A idx mod N
    }

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "NAS IS: %d keys (2^%.0f), MAX_KEY=%d, %d iterations",
            N, log2((double)N), max_key, MAX_ITERATIONS);
    if (klass) fprintf(stderr, "  [Class %c]", klass->cls);
    fprintf(stderr, "\n\n");

    size_t keysBytes = (size_t)N * sizeof(int);
    size_t rankBytes = (size_t)max_key * sizeof(int);

    // Kernel launch config
    const int bs        = block_size;
    const int tile      = 2 * bs;
    const int gridKeys  = (N + bs - 1) / bs;
    const int gridClear = (max_key + bs - 1) / bs;
    const int nScan     = (max_key + tile - 1) / tile;
    const size_t shmem  = (size_t)tile * sizeof(int);

    // Host allocation + key generation (untimed setup, upstream create_seq)
    int* h_keys    = (int*)malloc(keysBytes);   // mirror of device key_array
    int* ref_buff  = (int*)malloc(rankBytes);   // host-mirror rank array
    create_seq(h_keys, N, max_key, 314159265.0, 1220703125.0);

    // Device allocation
    int* d_key_array;
    int* d_key_buff1;   // rank array (histogram -> inclusive scan)
    int* d_key_buff2;   // copy of keys made during histogram (upstream)
    int* d_block_sums;
    HIP_CHECK(hipMalloc(&d_key_array, keysBytes));
    HIP_CHECK(hipMalloc(&d_key_buff1, rankBytes));
    HIP_CHECK(hipMalloc(&d_key_buff2, keysBytes));
    HIP_CHECK(hipMalloc(&d_block_sums, (size_t)nScan * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_key_array, h_keys, keysBytes, hipMemcpyHostToDevice));

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d", N, block_size);

    // Per-kernel accumulators across the 10 timed rank iterations.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_modify, acc_clear, acc_hist,
                acc_scanb, acc_scans, acc_scana;
    bench_lap_reset(&acc_modify);
    bench_lap_reset(&acc_clear);
    bench_lap_reset(&acc_hist);
    bench_lap_reset(&acc_scanb);
    bench_lap_reset(&acc_scans);
    bench_lap_reset(&acc_scana);

    // Launch each rank() kernel, timed (accumulating into its lap) or not.
    auto time_or_run = [&](bench_lap_t* acc, bool timed, auto launch) {
        if (timed) {
            bench_lap_add(acc, bench_time_op(&timer, launch));
        } else {
            launch();
            HIP_CHECK(hipGetLastError());
        }
    };

    // One rank(iteration): upstream's timed unit of work.
    auto run_rank = [&](int iteration, bool timed) {
        time_or_run(&acc_modify, timed, [&]() {
            modify_keys_kernel<<<dim3(1), dim3(1)>>>(d_key_array, iteration,
                                                     max_key);
        });
        time_or_run(&acc_clear, timed, [&]() {
            clear_kernel<<<dim3(gridClear), dim3(bs)>>>(d_key_buff1, max_key);
        });
        time_or_run(&acc_hist, timed, [&]() {
            histogram_kernel<<<dim3(gridKeys), dim3(bs)>>>(
                d_key_array, d_key_buff2, d_key_buff1, N);
        });
        time_or_run(&acc_scanb, timed, [&]() {
            scan_blocks_kernel<<<dim3(nScan), dim3(bs), shmem>>>(
                d_key_buff1, d_block_sums, max_key);
        });
        time_or_run(&acc_scans, timed, [&]() {
            scan_sums_kernel<<<dim3(1), dim3(bs), shmem>>>(d_block_sums, nScan);
        });
        time_or_run(&acc_scana, timed, [&]() {
            scan_add_kernel<<<dim3(nScan), dim3(bs)>>>(d_key_buff1,
                                                       d_block_sums, max_key);
        });
    };

    // Untimed rank(1): upstream's free iteration (warms data/code paths).
    run_rank(1, false);
    HIP_CHECK(hipDeviceSynchronize());

    /* Cold-cache calibration: evict the LLC (and hold the clock high) so the
     * timed run's first kernel launch runs cold. No-op when
     * BENCH_PARAM_vector_add_warmup is disabled. */
    bench_pass_warm();

    // Timed main loop: 10 rank iterations (upstream's timed section).
    // Partial verification per iteration runs on the host, untimed, against
    // the bit-exact host mirror (and published NPB tables at class sizes);
    // it reads back five 4-byte rank values per iteration.
    int pv_attempted = 0, pv_passed = 0;
    bool tables_ok = true;
    int pv_vals[TEST_ARRAY_SIZE];
    for (int iteration = 1; iteration <= MAX_ITERATIONS; ++iteration) {
        run_rank(iteration, true);

        if (!do_verify) continue;
        ref_rank(h_keys, ref_buff, N, max_key, iteration, test_index, pv_vals);
        for (int i = 0; i < TEST_ARRAY_SIZE; ++i) {
            int k = pv_vals[i];
            if (!(0 < k && k <= N - 1)) continue;   // upstream range guard
            pv_attempted++;
            int dev_rank;
            HIP_CHECK(hipMemcpy(&dev_rank, d_key_buff1 + (k - 1), sizeof(int),
                                hipMemcpyDeviceToHost));
            int expected = ref_buff[k - 1];
            if (dev_rank == expected) {
                pv_passed++;
            } else {
                fprintf(stderr,
                        "Failed partial verification: iteration %d, test key "
                        "%d (got %d, want %d)\n",
                        iteration, i, dev_rank, expected);
            }
            if (klass) {
                int pub = published_expected(klass->cls, i, iteration,
                                             klass->test_rank[i]);
                if (expected != pub) {
                    fprintf(stderr,
                            "Published-table mismatch: iteration %d, test key "
                            "%d (mirror %d, published %d)\n",
                            iteration, i, expected, pub);
                    tables_ok = false;
                }
            }
        }
    }

    double total_gpu_ms = acc_modify.gpu_ms + acc_clear.gpu_ms +
                          acc_hist.gpu_ms + acc_scanb.gpu_ms +
                          acc_scans.gpu_ms + acc_scana.gpu_ms;
    bench_emit_iter("modify_keys_kernel", 0, &acc_modify, params);
    bench_emit_iter("clear_kernel",       0, &acc_clear,  params);
    bench_emit_iter("histogram_kernel",   0, &acc_hist,   params);
    bench_emit_iter("scan_blocks_kernel", 0, &acc_scanb,  params);
    bench_emit_iter("scan_sums_kernel",   0, &acc_scans,  params);
    bench_emit_iter("scan_add_kernel",    0, &acc_scana,  params);
    fprintf(stderr,
            "modify gpu = %.3f ms, clear gpu = %.3f ms, histogram gpu = %.3f "
            "ms,\nscan_blocks gpu = %.3f ms, scan_sums gpu = %.3f ms, "
            "scan_add gpu = %.3f ms\n",
            acc_modify.gpu_ms, acc_clear.gpu_ms, acc_hist.gpu_ms,
            acc_scanb.gpu_ms, acc_scans.gpu_ms, acc_scana.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // full_verify (untimed, upstream): permute the keys with the device's
    // rank info and confirm sorted order. Additionally compare the device
    // rank array against the host mirror bit-for-bit (harness addition).
    // -------------------------------------------------------------------
    bool verify_ok = true;
    if (do_verify) {
        int* h_rank   = (int*)malloc(rankBytes);
        int* h_kb2    = (int*)malloc(keysBytes);
        int* h_sorted = (int*)malloc(keysBytes);
        HIP_CHECK(hipMemcpy(h_rank, d_key_buff1, rankBytes,
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_kb2, d_key_buff2, keysBytes,
                            hipMemcpyDeviceToHost));

        bool rank_match = (memcmp(h_rank, ref_buff, rankBytes) == 0);
        if (!rank_match)
            fprintf(stderr, "Device rank array differs from host mirror\n");

        // upstream full_verify: key_array[--key_buff_ptr[key_buff2[i]]] = key
        for (int i = 0; i < N; ++i)
            h_sorted[--h_rank[h_kb2[i]]] = h_kb2[i];
        int out_of_order = 0;
        for (int i = 1; i < N; ++i)
            if (h_sorted[i - 1] > h_sorted[i]) out_of_order++;
        bool full_ok = (out_of_order == 0);
        if (!full_ok)
            fprintf(stderr, "Full_verify: number of keys out of sort: %d\n",
                    out_of_order);

        // Upstream requires 5*MAX_ITERATIONS partial passes + full_verify.
        int pv_expected = klass ? TEST_ARRAY_SIZE * MAX_ITERATIONS
                                : pv_attempted;
        verify_ok = (pv_passed == pv_attempted) &&
                    (pv_attempted == pv_expected) &&
                    tables_ok && full_ok && rank_match;
        fprintf(stderr,
                "Verification: %d/%d partial + full_verify %s + rank array "
                "%s => %s\n",
                pv_passed, pv_expected, full_ok ? "ok" : "FAILED",
                rank_match ? "ok" : "FAILED", verify_ok ? "PASS" : "FAIL");

        free(h_rank);
        free(h_kb2);
        free(h_sorted);
    }

    HIP_CHECK(hipFree(d_key_array));
    HIP_CHECK(hipFree(d_key_buff1));
    HIP_CHECK(hipFree(d_key_buff2));
    HIP_CHECK(hipFree(d_block_sums));
    free(h_keys);
    free(ref_buff);

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
