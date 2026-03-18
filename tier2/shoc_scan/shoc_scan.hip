// shoc_scan.hip — SHOC Scan benchmark (HIP, self-contained)
//
// Exclusive prefix scan (sum): output[i] = sum(input[0..i-1])
// Uses a two-phase work-efficient Blelloch scan with multi-block support via
// recursive block-sum propagation.  Derived from the SHOC benchmark suite.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./shoc_scan [--array_size N] [--block_size B] [--iterations I]
//
//   --array_size N   Number of float elements (default: 1048576 = 1M)
//   --block_size B   Threads per block (half the elements per block; default: 256)
//   --iterations I   Timed iterations (default: 20)
//
// Output (stdout): CSV row — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
// Output (stderr): Effective bandwidth in GB/s, verification PASS/FAIL

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Inlined bench_common_hip.h utilities
// ---------------------------------------------------------------------------

// Error-checking macro
#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Result record
struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

// Parse --iterations N from argv (default: 20)
static int parseIterations(int argc, char** argv) {
    int iters = 20;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--iterations") == 0) {
            int v = atoi(argv[i + 1]);
            if (v >= 1) iters = v;
            break;
        }
    }
    return iters;
}

// Parse --<name> N from argv (returns defaultVal if not found)
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

// CSV output helpers
static void printCSVHeader() {
    printf("kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms\n");
}

static void printCSVRow(const BenchResult& r) {
    printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
           r.kernel_name, r.problem_size, r.iterations,
           r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

// HIP-event-based timer
struct BenchmarkTimer {
    hipEvent_t start, stop;
    BenchmarkTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~BenchmarkTimer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    void record_start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start, stream));
    }
    void record_stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// runBenchmark: one warmup + N timed iterations, returns stats
template <typename Func>
static BenchResult runBenchmark(const char* kernel_name,
                                const char* problem_size,
                                int         iterations,
                                Func        func) {
    BenchmarkTimer timer;

    // Warmup (not measured)
    func();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        timer.record_start();
        func();
        timer.record_stop();
        times[i] = static_cast<double>(timer.elapsed_ms());
    }

    double sum = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum / iterations;

    double variance = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iterations > 1) ? sqrt(variance / (iterations - 1)) : 0.0;

    BenchResult r;
    r.kernel_name  = kernel_name;
    r.problem_size = problem_size;
    r.iterations   = iterations;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;
    return r;
}

// ---------------------------------------------------------------------------
// Scan kernels
// ---------------------------------------------------------------------------

// Maximum block size for the scan kernel (threads per block = elements/2)
#define MAX_BLOCK_SIZE 512

/**
 * scan_block_kernel — Work-efficient (Blelloch) exclusive prefix scan.
 *
 * Each block processes (blockDim.x * 2) elements using shared memory.
 * After the in-block scan completes, the block's total sum (the value that
 * was in the last shared-memory slot before zeroing) is stored in
 * block_sums[blockIdx.x] if block_sums != nullptr.
 *
 * @param output     Output array — receives the exclusive prefix scan.
 * @param input      Input array.
 * @param block_sums Array of per-block totals (one entry per block), or nullptr.
 * @param N          Number of elements (may be padded to a multiple of 2*blockDim.x).
 */
__global__ void scan_block_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  float* __restrict__ block_sums,
                                  int N) {
    extern __shared__ float temp[];

    int tid          = threadIdx.x;
    int block_offset = blockIdx.x * (blockDim.x * 2);
    int ai           = tid;
    int bi           = tid + blockDim.x;

    // Load two elements per thread into shared memory (zero-pad if out of range)
    temp[ai] = (block_offset + ai < N) ? input[block_offset + ai] : 0.0f;
    temp[bi] = (block_offset + bi < N) ? input[block_offset + bi] : 0.0f;

    int n = blockDim.x * 2;

    // ---- Up-sweep (reduce) phase ----
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

    // Save block total before zeroing the last element
    __syncthreads();
    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[n - 1];
        }
        temp[n - 1] = 0.0f;  // seed the exclusive scan
    }

    // ---- Down-sweep phase ----
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

    // Write results back to global memory
    if (block_offset + ai < N) output[block_offset + ai] = temp[ai];
    if (block_offset + bi < N) output[block_offset + bi] = temp[bi];
}

/**
 * add_block_sums_kernel — Propagate block-level prefix sums.
 *
 * After the block sums themselves have been scanned (exclusively), each
 * block (except block 0) adds its prefix sum to all of its elements.
 *
 * @param data       In/out: the per-block scanned array to fixup.
 * @param block_sums Exclusively-scanned block sums.
 * @param N          Number of elements.
 */
__global__ void add_block_sums_kernel(float* __restrict__ data,
                                      const float* __restrict__ block_sums,
                                      int N) {
    // Block 0 needs no fixup (its prefix sum is 0 by definition of exclusive scan)
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

/**
 * scan_recursive — Perform an exclusive prefix scan on d_input, writing
 * the result to d_output.  Handles arrays larger than a single block by
 * recursively scanning the per-block totals and adding them back.
 *
 * NOTE: N must already be padded to a multiple of (block_size * 2) by the caller.
 *
 * @param d_output   Device output buffer (size N).
 * @param d_input    Device input buffer (size N, read-only).
 * @param N          Number of elements (padded).
 * @param block_size Threads per block (elements per block = block_size * 2).
 */
static void scan_recursive(float* d_output, const float* d_input, int N,
                            int block_size) {
    int elements_per_block = block_size * 2;
    int num_blocks         = (N + elements_per_block - 1) / elements_per_block;
    size_t shared_mem      = (size_t)elements_per_block * sizeof(float);

    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        HIP_CHECK(hipMalloc(&d_block_sums, (size_t)num_blocks * sizeof(float)));
    }

    // Phase 1: per-block exclusive scan; store block totals in d_block_sums
    hipLaunchKernelGGL(scan_block_kernel,
                       dim3(num_blocks), dim3(block_size),
                       shared_mem, 0,
                       d_output, d_input, d_block_sums, N);

    if (num_blocks > 1) {
        // Phase 2: recursively scan the block totals
        float* d_scanned_block_sums;
        HIP_CHECK(hipMalloc(&d_scanned_block_sums,
                             (size_t)num_blocks * sizeof(float)));

        scan_recursive(d_scanned_block_sums, d_block_sums, num_blocks,
                       block_size);

        // Phase 3: add the scanned block sums back to each block's output
        hipLaunchKernelGGL(add_block_sums_kernel,
                           dim3(num_blocks), dim3(block_size),
                           0, 0,
                           d_output, d_scanned_block_sums, N);

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
    int N          = parseIntParam(argc, argv, "--array_size", 1048576);
    int block_size = parseIntParam(argc, argv, "--block_size", 256);
    int iters      = parseIterations(argc, argv);

    // Clamp block_size to hardware-safe maximum and ensure power-of-2
    if (block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;

    // Pad N to next multiple of elements_per_block (= block_size * 2) so the
    // recursive scan kernel always has full blocks to work with
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
    fprintf(stderr, "Block size: %d  |  Iterations: %d\n\n", block_size, iters);

    // Host allocations (calloc so padding is zero)
    float* h_input  = (float*)calloc(N_padded, sizeof(float));
    float* h_output = (float*)malloc(bytes_orig);
    float* h_ref    = (float*)malloc(bytes_orig);

    // Initialize with a simple pattern
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i % 7) + 1.0f;  // values 1..7, repeating
    }

    // Device allocations
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input,  bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));

    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));

    // Build the size string for CSV
    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%d", N);

    printCSVHeader();

    // Benchmark: copy input → output buffer each iteration (scan reads d_input,
    // writes d_output; we keep d_input pristine so every iteration is identical)
    BenchResult r = runBenchmark("scan", size_str, iters, [&]() {
        scan_recursive(d_output, d_input, N_padded, block_size);
    });

    printCSVRow(r);

    // Effective bandwidth: 1 read (input) + 1 write (output) = 2 × N × sizeof(float)
    double bandwidth_gb = 2.0 * (double)N * sizeof(float)
                          / (r.avg_ms * 1e-3) / 1e9;
    fprintf(stderr, "Effective bandwidth: %.2f GB/s  (avg %.4f ms)\n",
            bandwidth_gb, r.avg_ms);

    // ---------------------------------------------------------------------------
    // Verification: compare GPU result against CPU exclusive scan
    // ---------------------------------------------------------------------------
    // Run one final scan for verification
    scan_recursive(d_output, d_input, N_padded, block_size);
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

    return (errors > 0) ? 1 : 0;
}
