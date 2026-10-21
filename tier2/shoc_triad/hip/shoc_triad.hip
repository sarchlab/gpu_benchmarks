// shoc_triad.hip — SHOC Triad benchmark (HIP, self-contained)
//
// Stream Triad: a[i] = b[i] + scalar * c[i]
// Classic memory bandwidth benchmark derived from the SHOC suite.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback:
//   BENCH_PARAM_array_size, BENCH_PARAM_block_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// launch carrying the GPU event time and host wall-clock time, then a summary
// whose total_time_ms equals that record's GPU event time. Effective
// bandwidth is derived downstream from time_ms and params.
// Output (stderr): Human-readable diagnostics

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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

// ---------------------------------------------------------------------------
// Triad kernel: a[i] = b[i] + scalar * c[i]
// ---------------------------------------------------------------------------
__global__ void triad_kernel(float* __restrict__ a,
                              const float* __restrict__ b,
                              const float* __restrict__ c,
                              float scalar, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = b[i] + scalar * c[i];
    }
}

// ---------------------------------------------------------------------------
// CPU reference for verification
// ---------------------------------------------------------------------------
static void triad_cpu(float* a, const float* b, const float* c,
                      float scalar, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = b[i] + scalar * c[i];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int array_size  = 4194304;
    int block_size  = 256;

    // Command-line fallback
    array_size = parseIntParam(argc, argv, "--array_size", array_size);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_array_size");
    if (env_val) array_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int N = array_size;
    size_t bytes  = (size_t)N * sizeof(float);
    float  scalar = 1.75f;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d floats (%.1f MiB)\n",
            N, (double)bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Block size: %d\n\n", block_size);

    // Host allocations
    float* h_a   = (float*)malloc(bytes);
    float* h_b   = (float*)malloc(bytes);
    float* h_c   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    // Initialize inputs
    for (int i = 0; i < N; ++i) {
        h_b[i] = (float)(i % 1000) * 0.001f;
        h_c[i] = (float)((i + 37) % 1000) * 0.001f;
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, h_c, bytes, hipMemcpyHostToDevice));

    int grid = (N + block_size - 1) / block_size;

    char params[128];
    snprintf(params, sizeof(params),
             "\"array_size\":%d,\"block_size\":%d",
             array_size, block_size);

    // Single timed launch (bench_run_once: optional cold-cache warmer, then
    // one bracketed launch). The kernel only reads b and c and overwrites a
    // from scratch, so every execution performs identical work — no state
    // reset needed.
    double total_gpu_ms =
        bench_run_once("triad_kernel", params, [&]() {
            triad_kernel<<<dim3(grid), dim3(block_size)>>>(d_a, d_b, d_c, scalar, N);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify correctness
    HIP_CHECK(hipMemcpy(h_a, d_a, bytes, hipMemcpyDeviceToHost));
    triad_cpu(h_ref, h_b, h_c, scalar, N);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_a[i] - h_ref[i]);
        if (!std::isfinite(h_a[i]) || !std::isfinite(h_ref[i]) ||
            diff > 1e-5f * fabsf(h_ref[i]) + 1e-6f) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: got %.8f, expected %.8f\n",
                        i, h_a[i], h_ref[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        fprintf(stderr, "FAIL: %d mismatches\n", errors);
    } else {
        fprintf(stderr, "PASS\n");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    return (errors > 0) ? 1 : 0;
}
