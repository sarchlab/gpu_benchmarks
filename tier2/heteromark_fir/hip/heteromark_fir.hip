// heteromark_fir.hip — HeteroMark FIR filter benchmark (HIP)
//
// 1D FIR (Finite Impulse Response) convolution:
//   output[i] = sum(coeff[k] * input[i - k]) for k = 0..num_taps-1,
//   with zero input history for i-k < 0.
//
// Each thread computes one output sample. Filter coefficients are loaded
// into shared memory for performance.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./heteromark_fir [--size N]
//
//   --size N         Number of input samples (default: 1048576)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_num_taps, BENCH_PARAM_block_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// fir_filter_kernel launch carrying the GPU event time (time_ms) and host
// wall-clock time (host_time_ms), then a summary whose total_time_ms is the
// sum of all recorded GPU event times. Throughput (GB/s) is derived
// downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
// Constants
// ---------------------------------------------------------------------------

#define NUM_TAPS 128

// ---------------------------------------------------------------------------
// Kernel: FIR filter with shared memory for coefficients
//   Each thread computes one output sample.
//   Coefficients are loaded into shared memory once per block.
// ---------------------------------------------------------------------------

__global__ void fir_filter_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   const float* __restrict__ coeff,
                                   int num_samples,
                                   int num_taps) {
    __shared__ float s_coeff[NUM_TAPS];

    // Cooperatively load coefficients into shared memory
    for (int t = threadIdx.x; t < num_taps; t += blockDim.x) {
        s_coeff[t] = coeff[t];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float sum = 0.0f;
    for (int k = 0; k < num_taps; ++k) {
        int in_idx = idx - k;
        if (in_idx >= 0) {
            sum += s_coeff[k] * input[in_idx];
        }
    }

    output[idx] = sum;
}

// ---------------------------------------------------------------------------
// CPU reference: FIR filter
// ---------------------------------------------------------------------------

static void fir_cpu(const float* input, float* output, const float* coeff,
                     int num_samples, int num_taps) {
    for (int i = 0; i < num_samples; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < num_taps; ++k) {
            int in_idx = i - k;
            if (in_idx >= 0) {
                sum += coeff[k] * input[in_idx];
            }
        }
        output[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int size       = 1048576;
    int num_taps   = 64;
    int block_size = 256;

    // Command-line fallback
    size       = parseIntParam(argc, argv, "--size", size);
    num_taps   = parseIntParam(argc, argv, "--num_taps", num_taps);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_taps");
    if (env_val) num_taps = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int N     = size;
    int taps  = num_taps;

    // The kernel stages coefficients into s_coeff[NUM_TAPS]; more taps than that
    // would overflow shared memory. Fail cleanly rather than corrupt memory.
    if (taps < 1 || taps > NUM_TAPS) {
        fprintf(stderr,
                "ERROR: num_taps=%d out of range (1..%d = shared-memory "
                "capacity NUM_TAPS). Rebuild with a larger NUM_TAPS to exceed it.\n",
                taps, NUM_TAPS);
        return EXIT_FAILURE;
    }

    size_t input_bytes  = (size_t)N * sizeof(float);
    size_t output_bytes = (size_t)N * sizeof(float);
    size_t coeff_bytes  = (size_t)taps * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "FIR filter  |  Samples: %d  |  Taps: %d\n\n",
            N, taps);

    // Host allocation + initialization
    float* h_input  = (float*)malloc(input_bytes);
    float* h_output = (float*)malloc(output_bytes);
    float* h_coeff  = (float*)malloc(coeff_bytes);

    // Initialize input with a simple pattern
    for (int i = 0; i < N; ++i) {
        h_input[i] = sinf((float)i * 0.01f) + 0.5f;
    }

    // Initialize filter coefficients (simple low-pass-like)
    float coeff_sum = 0.0f;
    for (int k = 0; k < taps; ++k) {
        h_coeff[k] = 1.0f / (float)(k + 1);
        coeff_sum += h_coeff[k];
    }
    // Normalize coefficients
    for (int k = 0; k < taps; ++k) {
        h_coeff[k] /= coeff_sum;
    }

    // Device allocation
    float* d_input;
    float* d_output;
    float* d_coeff;
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_output, output_bytes));
    HIP_CHECK(hipMalloc(&d_coeff, coeff_bytes));

    // Copy input and coefficients to device
    HIP_CHECK(hipMemcpy(d_input, h_input, input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coeff, h_coeff, coeff_bytes, hipMemcpyHostToDevice));

    // Kernel launch config
    int blockSize = block_size;
    int gridSize  = (N + blockSize - 1) / blockSize;

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"num_taps\":%d,\"block_size\":%d,"
             "\"verification\":\"full_cpu_v1\"",
             size, num_taps, block_size);

    // Single timed launch (bench_run_once). The kernel reads immutable
    // input/coeff arrays and fully overwrites the output array, so no
    // state reset is needed before the measurement.
    double total_gpu_ms =
        bench_run_once("fir_filter_kernel", params, [&]() {
            fir_filter_kernel<<<dim3(gridSize), dim3(blockSize)>>>(d_input, d_output, d_coeff, N, taps);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // Verification of every actual output from the measured launch
    // -------------------------------------------------------------------
    bool verify_ok = true;
    {
        HIP_CHECK(hipMemcpy(h_output, d_output, output_bytes, hipMemcpyDeviceToHost));

        float* h_ref = (float*)malloc(output_bytes);
        fir_cpu(h_input, h_ref, h_coeff, N, taps);

        int errors = 0;
        for (int i = 0; i < N; ++i) {
            float diff = fabsf(h_output[i] - h_ref[i]);
            float tol = 1e-4f * fabsf(h_ref[i]) + 1e-6f;
            if (!std::isfinite(h_output[i]) || !std::isfinite(h_ref[i]) ||
                diff > tol) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at %d: GPU=%.6f CPU=%.6f diff=%.2e\n",
                            i, h_output[i], h_ref[i], diff);
                }
                errors++;
            }
        }
        verify_ok = errors == 0;
        fprintf(stderr, "FIR verification: %s (%d samples, %d mismatches)\n",
                verify_ok ? "PASS" : "FAIL", N, errors);

        free(h_ref);
    }

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_coeff));
    free(h_input);
    free(h_output);
    free(h_coeff);

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
