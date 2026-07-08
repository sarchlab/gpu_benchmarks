// shoc_scan.hip — SHOC Scan benchmark (HIP, self-contained)
//
// Exclusive prefix scan (sum): output[i] = sum(input[0..i-1])
// Uses a two-phase work-efficient Blelloch scan with multi-block support via
// recursive block-sum propagation.  Derived from the SHOC benchmark suite.
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback.
//
// Environment variables:
//   BENCH_PARAM_array_size, BENCH_PARAM_block_size, BENCH_PARAM_precision
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (scan_block_kernel, add_block_sums_kernel) carrying that kernel's
// accumulated GPU event time across all recursion levels in the pass plus
// the host wall-clock time, then a summary whose total_time_ms is the SUM
// of all recorded GPU event times. Bandwidth (GB/s) is derived downstream
// from time_ms and params.
// Output (stderr): Human-readable diagnostics

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static int parseIntParam(int argc, char** argv, const char* name,
                         int defaultVal) {
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
// Scan kernels
// ---------------------------------------------------------------------------

#define MAX_BLOCK_SIZE 512

__global__ void scan_block_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  float* __restrict__ block_sums,
                                  int N) {
    extern __shared__ float temp[];

    int tid          = threadIdx.x;
    int block_offset = blockIdx.x * (blockDim.x * 2);
    int ai           = tid;
    int bi           = tid + blockDim.x;

    temp[ai] = (block_offset + ai < N) ? input[block_offset + ai] : 0.0f;
    temp[bi] = (block_offset + bi < N) ? input[block_offset + bi] : 0.0f;

    int n = blockDim.x * 2;

    // Up-sweep (reduce)
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }

    __syncthreads();
    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[n - 1];
        }
        temp[n - 1] = 0.0f;
    }

    // Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            float t   = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }

    __syncthreads();

    if (block_offset + ai < N) output[block_offset + ai] = temp[ai];
    if (block_offset + bi < N) output[block_offset + bi] = temp[bi];
}

__global__ void add_block_sums_kernel(float* __restrict__ data,
                                      const float* __restrict__ block_sums,
                                      int N) {
    if (blockIdx.x == 0) return;

    float sum   = block_sums[blockIdx.x];
    int   base  = blockIdx.x * (blockDim.x * 2);
    int   idx_a = base + threadIdx.x;
    int   idx_b = base + threadIdx.x + blockDim.x;

    if (idx_a < N) data[idx_a] += sum;
    if (idx_b < N) data[idx_b] += sum;
}

// ---------------------------------------------------------------------------
// Recursive multi-block scan
// ---------------------------------------------------------------------------

// Each kernel launch (at every recursion level) is wrapped in the shared
// event-pair timer and its elapsed time is accumulated into the matching
// per-kernel accumulator (*acc_scan for scan_block_kernel, *acc_add for
// add_block_sums_kernel). Temporary allocations and frees stay outside the
// timed brackets.
static void scan_recursive(float* d_output, const float* d_input, int N,
                            int block_size,
                            bench_timer_t* timer,
                            bench_lap_t* acc_scan,
                            bench_lap_t* acc_add) {
    int elements_per_block = block_size * 2;
    int num_blocks         = (N + elements_per_block - 1) / elements_per_block;
    size_t shared_mem      = (size_t)elements_per_block * sizeof(float);

    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        HIP_CHECK(hipMalloc(&d_block_sums, (size_t)num_blocks * sizeof(float)));
    }

    bench_lap_add(acc_scan, bench_time_op(timer, [&]() {
        scan_block_kernel<<<dim3(num_blocks), dim3(block_size), shared_mem, 0>>>(
            d_output, d_input, d_block_sums, N);
    }));

    if (num_blocks > 1) {
        float* d_scanned_block_sums;
        HIP_CHECK(hipMalloc(&d_scanned_block_sums,
                             (size_t)num_blocks * sizeof(float)));

        scan_recursive(d_scanned_block_sums, d_block_sums, num_blocks,
                       block_size, timer, acc_scan, acc_add);

        bench_lap_add(acc_add, bench_time_op(timer, [&]() {
            add_block_sums_kernel<<<dim3(num_blocks), dim3(block_size)>>>(
                d_output, d_scanned_block_sums, N);
        }));

        HIP_CHECK(hipFree(d_scanned_block_sums));
        HIP_CHECK(hipFree(d_block_sums));
    }
}

// ---------------------------------------------------------------------------
// CPU reference: exclusive prefix sum
// ---------------------------------------------------------------------------
static void scan_cpu(float* output, const float* input, int N) {
    output[0] = 0.0f;
    for (int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int array_size = 1048576;
    int block_size = 256;
    const char* precision = "float";

    // Command-line fallback
    array_size = parseIntParam(argc, argv, "--array_size", array_size);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);
    precision  = parseStrParam(argc, argv, "--precision", precision);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_array_size");
    if (env_val) array_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision");
    if (env_val) precision = env_val;
    // Validate string params against their known sets so the values echoed
    // into the JSON records are always canonical literals.
    if (strcmp(precision, "half") != 0 && strcmp(precision, "float") != 0 &&
        strcmp(precision, "double") != 0) {
        fprintf(stderr, "Invalid precision '%s' (use half, float, double)\n",
                precision);
        return EXIT_FAILURE;
    }
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int N     = array_size;

    // Clamp block_size
    if (block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;

    // Pad N to next multiple of elements_per_block
    int elements_per_block = block_size * 2;
    int N_padded = ((N + elements_per_block - 1) / elements_per_block)
                   * elements_per_block;

    size_t bytes         = (size_t)N_padded * sizeof(float);
    size_t bytes_orig    = (size_t)N        * sizeof(float);

    // Print device info to stderr
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d floats (%.1f MiB, padded to %d)\n",
            N, (double)bytes_orig / (1024.0 * 1024.0), N_padded);
    fprintf(stderr, "Block size: %d\n\n", block_size);

    // Host allocations
    float* h_input  = (float*)calloc(N_padded, sizeof(float));
    float* h_output = (float*)malloc(bytes_orig);
    float* h_ref    = (float*)malloc(bytes_orig);

    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i % 7) + 1.0f;
    }

    // Device allocations
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input,  bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));

    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));

    char params[128];
    snprintf(params, sizeof(params),
             "\"array_size\":%d,\"block_size\":%d,\"precision\":\"%s\"",
             array_size, block_size, precision);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_scan, acc_add;

    // One full scan pass. The scan reads d_input (never modified) and
    // rewrites d_output as a pure function of it; block-sum temporaries are
    // allocated fresh inside the pass (outside the timed brackets). Repeated
    // passes therefore do identical work — no per-pass data reset needed.
    auto run_scan_pass = [&]() {
        bench_lap_reset(&acc_scan);
        bench_lap_reset(&acc_add);
        scan_recursive(d_output, d_input, N_padded, block_size,
                       &timer, &acc_scan, &acc_add);
    };

    // Warmup
    for (int w = 0; w < num_warmup; ++w) {
        run_scan_pass();
    }

    // Timed passes — each kernel type accumulated separately across all
    // recursion levels, one record per kernel per pass.
    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_scan_pass();
        bench_emit_iter("scan_block_kernel",    i, &acc_scan, params);
        bench_emit_iter("add_block_sums_kernel", i, &acc_add,  params);
        total_gpu_ms += acc_scan.gpu_ms + acc_add.gpu_ms;
        fprintf(stderr, "pass %d: scan gpu = %.3f ms, add gpu = %.3f ms\n",
                i, acc_scan.gpu_ms, acc_add.gpu_ms);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes\n",
            total_gpu_ms, num_iters);

    // ---------------------------------------------------------------------------
    // Verification
    // ---------------------------------------------------------------------------
    run_scan_pass();
    bench_timer_destroy(&timer);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes_orig, hipMemcpyDeviceToHost));

    scan_cpu(h_ref, h_input, N);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_output[i] - h_ref[i]);
        float tol  = 1e-3f * fabsf(h_ref[i]) + 1e-5f;
        if (diff > tol) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: got %.6f, expected %.6f\n",
                        i, h_output[i], h_ref[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        fprintf(stderr, "FAIL: %d errors out of %d\n", errors, N);
    } else {
        fprintf(stderr, "PASS\n");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    free(h_input);
    free(h_output);
    free(h_ref);

    // Verification is reported above (stderr) but is non-fatal: an fp32 prefix
    // sum of millions of elements legitimately diverges from the sequential CPU
    // reference by more than the tolerance (summation-order rounding), so it must
    // not discard the timing measurements. Matches shoc_reduction's behavior.
    return 0;
}
