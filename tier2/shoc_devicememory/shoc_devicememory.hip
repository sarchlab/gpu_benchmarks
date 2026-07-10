// shoc_devicememory.hip — SHOC DeviceMemory benchmark (HIP, self-contained)
//
// Device global memory bandwidth benchmark: read-only, write-only, and
// read-write patterns.
//
// Parameters are read from BENCH_PARAM_* environment variables,
// with command-line --flags as fallback:
//   BENCH_PARAM_array_size_mb, BENCH_PARAM_block_size,
//   BENCH_PARAM_transfer_mode (read | write | readwrite)
//   BENCH_PARAM_warmup   (warmup launches per kernel, default 3)
//   BENCH_PARAM_iteration    (timed launches per kernel, each recorded, default 7)
//
// Output (stdout): per enabled kernel, one {"type":"kernel",...} JSON line per
// timed iteration carrying the GPU event time and host wall-clock time, then a
// summary whose total_time_ms is the SUM of all recorded GPU event times.
// GB/s figures are derived downstream from time_ms and params.
// Output (stderr): Human-readable diagnostics

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
// Kernels
// ---------------------------------------------------------------------------

__global__ void readOnly_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = gid; i < N; i += totalThreads) {
        sum += input[i];
    }
    if (gid < N) {
        output[gid] = sum;
    }
}

__global__ void writeOnly_kernel(float* __restrict__ output,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        output[i] = (float)i * 0.001f;
    }
}

__global__ void readWrite_kernel(float* __restrict__ data,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        data[i] = data[i] * 1.0001f;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int array_size_mb = 64;
    int block_size    = 256;
    const char* transfer_mode = "readwrite";

    // Command-line fallback
    array_size_mb = parseIntParam(argc, argv, "--array_size_mb", array_size_mb);
    block_size    = parseIntParam(argc, argv, "--block_size", block_size);
    transfer_mode = parseStrParam(argc, argv, "--transfer_mode", transfer_mode);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_array_size_mb");
    if (env_val) array_size_mb = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_transfer_mode");
    if (env_val) transfer_mode = env_val;
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    // Validate transfer_mode against its known set and canonicalize it so the
    // params JSON never echoes raw env/argv text.
    if (strcmp(transfer_mode, "read") == 0) {
        transfer_mode = "read";
    } else if (strcmp(transfer_mode, "write") == 0) {
        transfer_mode = "write";
    } else if (strcmp(transfer_mode, "readwrite") == 0) {
        transfer_mode = "readwrite";
    } else {
        fprintf(stderr, "Unsupported transfer_mode '%s' (supported: read, write, readwrite)\n",
                transfer_mode);
        return EXIT_FAILURE;
    }

    size_t N     = (size_t)array_size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Array size: %d MB (%zu floats)  |  Block size: %d\n\n",
            array_size_mb, N, block_size);

    // Device allocations
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input,  bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));

    // Initialize device input
    float* h_init = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i)
        h_init[i] = (float)(i % 1000) * 0.001f;
    HIP_CHECK(hipMemcpy(d_input, h_init, bytes, hipMemcpyHostToDevice));
    free(h_init);

    int num_blocks = (int)((N + block_size - 1) / block_size);
    if (num_blocks > 65535) num_blocks = 65535;

    char params[160];
    snprintf(params, sizeof(params),
             "\"array_size_mb\":%d,\"block_size\":%d,\"transfer_mode\":\"%s\"",
             array_size_mb, block_size, transfer_mode);

    double total_gpu_ms = 0.0;

    // Determine which kernels to run based on transfer_mode
    bool run_read  = (strcmp(transfer_mode, "read") == 0 || strcmp(transfer_mode, "readwrite") == 0);
    bool run_write = (strcmp(transfer_mode, "write") == 0 || strcmp(transfer_mode, "readwrite") == 0);
    bool run_rw    = (strcmp(transfer_mode, "readwrite") == 0);

    // --- Read-only bandwidth ---
    if (run_read) {
        // readOnly_kernel only reads d_input and overwrites d_output;
        // repeated launches perform identical work.
        total_gpu_ms += bench_run_loop("readOnly_kernel", num_warmup, num_iters,
                                       params, [&]() {
            readOnly_kernel<<<dim3(num_blocks), dim3(block_size)>>>(d_input, d_output, (int)N);
        });
    }

    // --- Write-only bandwidth ---
    if (run_write) {
        // writeOnly_kernel overwrites d_output from scratch every launch.
        total_gpu_ms += bench_run_loop("writeOnly_kernel", num_warmup, num_iters,
                                       params, [&]() {
            writeOnly_kernel<<<dim3(num_blocks), dim3(block_size)>>>(d_output, (int)N);
        });
    }

    // --- Read-write bandwidth ---
    if (run_rw) {
        // Re-initialize d_input (readOnly warmups do not touch it, but keep
        // the pristine values for a well-defined starting point)
        {
            float* h_rw = (float*)malloc(bytes);
            for (size_t i = 0; i < N; ++i) h_rw[i] = (float)(i % 1000) * 0.001f;
            HIP_CHECK(hipMemcpy(d_input, h_rw, bytes, hipMemcpyHostToDevice));
            free(h_rw);
        }
        // readWrite_kernel scales data in place by 1.0001; the float multiply
        // is value-independent (no control flow or memory-access pattern
        // depends on the data), so no per-iteration reset is needed.
        total_gpu_ms += bench_run_loop("readWrite_kernel", num_warmup, num_iters,
                                       params, [&]() {
            readWrite_kernel<<<dim3(num_blocks), dim3(block_size)>>>(d_input, (int)N);
        });
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms (%d iterations per kernel)\n",
            total_gpu_ms, num_iters);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    return 0;
}
