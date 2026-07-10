// cuda_scan_large.hip — Large-array parallel prefix sum (Blelloch scan) benchmark (HIP)
//
// Work-efficient parallel prefix sum (exclusive scan) for large arrays.
// Three-phase hierarchical decomposition:
//   1. Block-level scan of each block
//   2. Scan of block sums
//   3. Add block sums back to each block
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./cuda_scan_large [--size N]
//
//   --size N         Number of elements (default: 16777216 = 16M)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size, BENCH_PARAM_verify
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (scan_block_kernel, add_block_sums_kernel) carrying that kernel's
// accumulated GPU event time and host time within the pass (summed over all
// recursion levels), then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. GB/s is derived downstream from time_ms + params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
// Constants
// ---------------------------------------------------------------------------

#define BLOCK_SIZE 256
#define ELEMENTS_PER_BLOCK (2 * BLOCK_SIZE)

// ---------------------------------------------------------------------------
// Kernel: Blelloch work-efficient scan (per-block)
//   Each block processes ELEMENTS_PER_BLOCK elements.
//   Stores the block total in block_sums[blockIdx.x] before zeroing the root.
// ---------------------------------------------------------------------------

__global__ void scan_block_kernel(
    unsigned int* __restrict__ data,
    unsigned int* __restrict__ block_sums,
    int N)
{
    __shared__ unsigned int temp[ELEMENTS_PER_BLOCK];

    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;

    // Load input into shared memory
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int ga = blockOffset + ai;
    int gb = blockOffset + bi;

    temp[ai] = (ga < N) ? data[ga] : 0;
    temp[bi] = (gb < N) ? data[gb] : 0;

    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }

    __syncthreads();

    // Store block sum and clear last element
    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[ELEMENTS_PER_BLOCK - 1];
        }
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }

    __syncthreads();

    // Write results back
    if (ga < N) data[ga] = temp[ai];
    if (gb < N) data[gb] = temp[bi];
}

// ---------------------------------------------------------------------------
// Kernel: add block sums to each element in the block
// ---------------------------------------------------------------------------

__global__ void add_block_sums_kernel(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ block_sums,
    int N)
{
    int idx = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x;
    unsigned int val = block_sums[blockIdx.x];

    if (idx < N)
        data[idx] += val;
    if (idx + BLOCK_SIZE < N)
        data[idx + BLOCK_SIZE] += val;
}

// ---------------------------------------------------------------------------
// Recursive scan for large arrays
// ---------------------------------------------------------------------------

// Each kernel launch is bracketed with its own event pair (bench_time_op) and
// summed across ALL recursion levels into the corresponding accumulator. The
// hipMalloc/hipFree of the block-sums buffer stay OUTSIDE the timed brackets.
static void scan_recursive(unsigned int* d_data, int N,
                           bench_timer_t* timer,
                           bench_lap_t* acc_block, bench_lap_t* acc_add) {
    int numBlocks = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    unsigned int* d_block_sums = nullptr;
    if (numBlocks > 1) {
        HIP_CHECK(hipMalloc(&d_block_sums, numBlocks * sizeof(unsigned int)));
    }

    // Phase 1: Block-level scan
    bench_lap_add(acc_block, bench_time_op(timer, [&]() {
        scan_block_kernel<<<dim3(numBlocks), dim3(BLOCK_SIZE)>>>(d_data, d_block_sums, N);
    }));

    if (numBlocks > 1) {
        // Phase 2: Recursively scan the block sums
        scan_recursive(d_block_sums, numBlocks, timer, acc_block, acc_add);

        // Phase 3: Add scanned block sums back to each block
        bench_lap_add(acc_add, bench_time_op(timer, [&]() {
            add_block_sums_kernel<<<dim3(numBlocks), dim3(BLOCK_SIZE)>>>(d_data, d_block_sums, N);
        }));

        HIP_CHECK(hipFree(d_block_sums));
    }
}

// ---------------------------------------------------------------------------
// CPU reference: exclusive prefix sum
// ---------------------------------------------------------------------------

static void prefix_sum_cpu(const unsigned int* input, unsigned int* output, int N) {
    unsigned int sum = 0;
    for (int i = 0; i < N; ++i) {
        output[i] = sum;
        sum += input[i];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static const char* parseStrParam(int argc, char** argv, const char* name,
                                 const char* defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return defaultVal;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int size       = 16777216;
    int block_size = 256;
    const char* verify = "true";

    // Command-line fallback
    size       = parseIntParam(argc, argv, "--size", size);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);
    verify     = parseStrParam(argc, argv, "--verify", verify);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_verify");
    if (env_val) verify = env_val;
    // Validate string params so the values echoed into the JSON records are
    // always canonical literals (never raw env text).
    if (strcmp(verify, "true") != 0 && strcmp(verify, "false") != 0 &&
        strcmp(verify, "skip") != 0) {
        fprintf(stderr, "Invalid verify '%s' (use true, false, skip)\n", verify);
        return EXIT_FAILURE;
    }

    int N     = size;

    size_t bytes = (size_t)N * sizeof(unsigned int);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Scan size: %d\n\n", N);

    // Host allocation + initialization
    unsigned int* h_input  = (unsigned int*)malloc(bytes);
    unsigned int* h_output = (unsigned int*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_input[i] = (unsigned int)(i % 10);  // small values to avoid overflow
    }

    // Device allocation
    unsigned int* d_data;
    HIP_CHECK(hipMalloc(&d_data, bytes));

    char params[160];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d,\"verify\":\"%s\"",
             size, block_size, verify);

    // Per-kernel accumulators: each pass runs one full hierarchical scan; the
    // accumulated GPU/host time across all recursion levels within one pass
    // becomes that pass's record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_block, acc_add;

    // One full scan pass. The scan is IN PLACE, so the pristine input is
    // re-uploaded at the start of every pass (untimed) — otherwise later
    // passes would scan already-scanned data.
    auto run_scan_pass = [&]() {
        bench_lap_reset(&acc_block);
        bench_lap_reset(&acc_add);
        HIP_CHECK(hipMemcpy(d_data, h_input, bytes, hipMemcpyHostToDevice));
        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();
        scan_recursive(d_data, N, &timer, &acc_block, &acc_add);
    };

    double total_gpu_ms = 0.0;
    run_scan_pass();
    bench_emit_iter("scan_block_kernel",     0, &acc_block, params);
    bench_emit_iter("add_block_sums_kernel", 0, &acc_add,   params);
    total_gpu_ms += acc_block.gpu_ms + acc_add.gpu_ms;
    fprintf(stderr, "scan_block gpu = %.3f ms, add_block_sums gpu = %.3f ms\n",
            acc_block.gpu_ms, acc_add.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------
    {
        HIP_CHECK(hipMemcpy(h_output, d_data, bytes, hipMemcpyDeviceToHost));

        // CPU reference
        unsigned int* h_ref = (unsigned int*)malloc(bytes);
        prefix_sum_cpu(h_input, h_ref, N);

        int errors = 0;
        for (int i = 0; i < N; ++i) {
            if (h_output[i] != h_ref[i]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at %d: GPU=%u CPU=%u\n",
                            i, h_output[i], h_ref[i]);
                }
                errors++;
            }
        }

        if (errors > 0)
            fprintf(stderr, "FAIL: %d errors out of %d\n", errors, N);
        else
            fprintf(stderr, "PASS\n");

        free(h_ref);
    }

    HIP_CHECK(hipFree(d_data));
    free(h_input);
    free(h_output);

    return 0;
}
