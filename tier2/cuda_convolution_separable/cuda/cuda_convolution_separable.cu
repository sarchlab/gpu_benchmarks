// cuda_convolution_separable.cu — Separable 2D convolution benchmark (CUDA)
//
// Two-pass separable 2D convolution: row pass + column pass.
//
// Native CUDA port of ../hip/cuda_convolution_separable.hip; compiles with nvcc.
//
// Usage:
//   ./cuda_convolution_separable [--width W] [--height H] [--radius R]
//
//   --width W         Image width  (default: 4096)
//   --height H        Image height (default: 4096)
//   --radius R        Kernel radius (default: 8, filter width = 2*R+1)
//
// Environment variables (override argv):
//   BENCH_PARAM_width, BENCH_PARAM_height, BENCH_PARAM_radius
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (convolution_row_kernel, convolution_col_kernel) carrying that
// kernel's GPU event time (time_ms) and host wall-clock time (host_time_ms)
// within the pass, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. Throughput (GB/s) is derived downstream from
// time_ms and params.
// Output (stderr): human-readable results

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <exception>

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

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

#define MAX_KERNEL_RADIUS 32
#define MAX_KERNEL_LENGTH (2 * MAX_KERNEL_RADIUS + 1)

// Convolution kernel stored in constant memory
__constant__ float c_kernel[MAX_KERNEL_LENGTH];

// ---------------------------------------------------------------------------
// Kernel: row convolution (horizontal pass)
//   Each thread computes one output pixel.
//   Uses shared memory to cache a row tile including halo.
// ---------------------------------------------------------------------------

#define ROW_TILE_W 128
#define BLOCK_ROWS 8

__global__ void convolution_row_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int radius)
{
    // Shared memory: row tile with halo on both sides
    extern __shared__ float smem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCK_ROWS + ty;
    int col = blockIdx.x * ROW_TILE_W + tx;

    int smem_width = ROW_TILE_W + 2 * radius;

    // Load tile + halo into shared memory
    // Each thread loads one center element + potentially halo elements
    int smem_col = tx + radius;
    if (row < height) {
        // Center
        smem[ty * smem_width + smem_col] =
            (col < width) ? input[row * width + col] : 0.0f;

        // Left halo
        if (tx < radius) {
            int halo_col = col - radius;
            smem[ty * smem_width + tx] =
                (halo_col >= 0 && row < height) ? input[row * width + halo_col] : 0.0f;
        }

        // Right halo
        if (tx >= ROW_TILE_W - radius) {
            int halo_col = col + radius;
            smem[ty * smem_width + smem_col + radius] =
                (halo_col < width && row < height) ? input[row * width + halo_col] : 0.0f;
        }
    } else {
        smem[ty * smem_width + smem_col] = 0.0f;
        if (tx < radius) smem[ty * smem_width + tx] = 0.0f;
        if (tx >= ROW_TILE_W - radius)
            smem[ty * smem_width + smem_col + radius] = 0.0f;
    }

    __syncthreads();

    // Compute convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += smem[ty * smem_width + smem_col + k] * c_kernel[radius + k];
        }
        output[row * width + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel: column convolution (vertical pass)
//   Each thread computes one output pixel.
//   Uses shared memory to cache a column tile including halo.
// ---------------------------------------------------------------------------

#define COL_TILE_W 16
// COL_TILE_W * COL_TILE_H is the column-kernel block size and must stay
// within the 1024 threads/block hardware limit: 16 * 64 = 1024.
#define COL_TILE_H 64

__global__ void convolution_col_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int radius)
{
    extern __shared__ float smem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * COL_TILE_W + tx;
    int row = blockIdx.y * COL_TILE_H + ty;

    int smem_height = COL_TILE_H + 2 * radius;

    int smem_row = ty + radius;

    // Load center
    if (col < width) {
        smem[smem_row * COL_TILE_W + tx] =
            (row < height) ? input[row * width + col] : 0.0f;

        // Top halo
        if (ty < radius) {
            int halo_row = row - radius;
            smem[ty * COL_TILE_W + tx] =
                (halo_row >= 0 && col < width) ? input[halo_row * width + col] : 0.0f;
        }

        // Bottom halo
        if (ty >= COL_TILE_H - radius) {
            int halo_row = row + radius;
            smem[(smem_row + radius) * COL_TILE_W + tx] =
                (halo_row < height && col < width) ? input[halo_row * width + col] : 0.0f;
        }
    } else {
        smem[smem_row * COL_TILE_W + tx] = 0.0f;
        if (ty < radius) smem[ty * COL_TILE_W + tx] = 0.0f;
        if (ty >= COL_TILE_H - radius)
            smem[(smem_row + radius) * COL_TILE_W + tx] = 0.0f;
    }

    __syncthreads();

    // Compute convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += smem[(smem_row + k) * COL_TILE_W + tx] * c_kernel[radius + k];
        }
        output[row * width + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: separable 2D convolution
// ---------------------------------------------------------------------------

static void conv_separable_cpu(const float* input, float* temp, float* output,
                               int width, int height,
                               const float* kernel_weights, int radius) {
    // Row pass
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int cc = c + k;
                float val = (cc >= 0 && cc < width) ? input[r * width + cc] : 0.0f;
                sum += val * kernel_weights[radius + k];
            }
            temp[r * width + c] = sum;
        }
    }

    // Column pass
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int rr = r + k;
                float val = (rr >= 0 && rr < height) ? temp[rr * width + c] : 0.0f;
                sum += val * kernel_weights[radius + k];
            }
            output[r * width + c] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int width  = 4096;
    int height = 4096;
    int radius = 8;

    // Command-line fallback
    width  = parseIntParam(argc, argv, "--width",  width);
    height = parseIntParam(argc, argv, "--height", height);
    radius = parseIntParam(argc, argv, "--radius", radius);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_width");
    if (env_val) width = atoi(env_val);
    env_val = getenv("BENCH_PARAM_height");
    if (env_val) height = atoi(env_val);
    env_val = getenv("BENCH_PARAM_radius");
    if (env_val) radius = atoi(env_val);

    if (radius > MAX_KERNEL_RADIUS) {
        fprintf(stderr, "Error: radius=%d exceeds max=%d\n", radius, MAX_KERNEL_RADIUS);
        return 1;
    }

    int klen = 2 * radius + 1;
    size_t imageBytes = (size_t)width * height * sizeof(float);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Image: %dx%d  |  Kernel radius: %d (%d-tap)\n\n",
            width, height, radius, klen);

    // Generate Gaussian-like kernel weights
    std::vector<float> h_kernel((size_t)klen);
    {
        float sigma = (float)radius / 3.0f;
        float sum = 0.0f;
        for (int i = 0; i < klen; ++i) {
            float x = (float)(i - radius);
            h_kernel[i] = expf(-x * x / (2.0f * sigma * sigma));
            sum += h_kernel[i];
        }
        for (int i = 0; i < klen; ++i) h_kernel[i] /= sum;
    }

    // Copy kernel to constant memory (setup — outside all timed brackets)
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel.data(),
                                 klen * sizeof(float)));

    // Host allocation + initialization
    float* h_input  = (float*)malloc(imageBytes);
    float* h_output = (float*)malloc(imageBytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "FAIL: host image buffers unavailable\n");
        free(h_input);
        free(h_output);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = (float)(i % 256) / 255.0f;
    }

    // Device allocation
    float *d_input, *d_temp, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  imageBytes));
    CUDA_CHECK(cudaMalloc(&d_temp,   imageBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imageBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration for row pass
    dim3 rowBlock(ROW_TILE_W, BLOCK_ROWS);
    dim3 rowGrid((width + ROW_TILE_W - 1) / ROW_TILE_W,
                 (height + BLOCK_ROWS - 1) / BLOCK_ROWS);
    size_t rowSmem = BLOCK_ROWS * (ROW_TILE_W + 2 * radius) * sizeof(float);

    // Kernel launch configuration for column pass
    dim3 colBlock(COL_TILE_W, COL_TILE_H);
    dim3 colGrid((width + COL_TILE_W - 1) / COL_TILE_W,
                 (height + COL_TILE_H - 1) / COL_TILE_H);
    size_t colSmem = COL_TILE_W * (COL_TILE_H + 2 * radius) * sizeof(float);

    char params[128];
    snprintf(params, sizeof(params),
             "\"width\":%d,\"height\":%d,\"radius\":%d,\"verification\":\"full_cpu_v1\"",
             width, height, radius);

    // Per-kernel accumulators: each pass runs the row kernel then the
    // column kernel once, each launch individually bracketed.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_row, acc_col;

    // One full separable-convolution pass. The row kernel reads the
    // never-mutated d_input and overwrites d_temp; the column kernel reads
    // d_temp and overwrites d_output. No pass mutates its own inputs, so
    // every pass performs identical work — no state reset is needed.
    auto run_conv_pass = [&]() {
        bench_lap_reset(&acc_row);
        bench_lap_reset(&acc_col);

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_row, bench_time_op(&timer, [&]() {
            convolution_row_kernel<<<rowGrid, rowBlock, rowSmem, 0>>>(
                d_input, d_temp, width, height, radius);
        }));

        bench_lap_add(&acc_col, bench_time_op(&timer, [&]() {
            convolution_col_kernel<<<colGrid, colBlock, colSmem, 0>>>(
                d_temp, d_output, width, height, radius);
        }));
    };

    double total_gpu_ms = 0.0;
    run_conv_pass();
    bench_emit_iter("convolution_row_kernel", 0, &acc_row, params);
    bench_emit_iter("convolution_col_kernel", 0, &acc_col, params);
    total_gpu_ms += acc_row.gpu_ms + acc_col.gpu_ms;
    fprintf(stderr, "row gpu = %.3f ms, col gpu = %.3f ms\n",
            acc_row.gpu_ms, acc_col.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // Verify both measured stages after all timing, using the original
    // full-image CPU arithmetic and tolerance. No GPU rerun or sampled fallback.
    // -------------------------------------------------------------------
    bool verify_ok = false;
    try {
        const size_t count = (size_t)width * height;
        std::vector<float> row_ref(count), ref(count);
        conv_separable_cpu(h_input, row_ref.data(), ref.data(), width, height,
                           h_kernel.data(), radius);

        size_t errors = 0;
        auto compare = [&](const char* stage, const float* device,
                           std::vector<float>& ref) {
            CUDA_CHECK(cudaMemcpy(h_output, device, imageBytes, cudaMemcpyDeviceToHost));
            for (size_t idx = 0; idx < count; ++idx) {
                float diff = fabsf(h_output[idx] - ref[idx]);
                float tol = 1e-3f * fabsf(ref[idx]) + 1e-5f;
                if (!std::isfinite(h_output[idx]) || !std::isfinite(ref[idx]) ||
                    diff > tol) {
                    if (errors < 10) {
                        fprintf(stderr, "Mismatch %s at (%zu,%zu): GPU=%.6f CPU=%.6f diff=%.6f\n",
                                stage, idx / width, idx % width,
                                h_output[idx], ref[idx], diff);
                    }
                    ++errors;
                }
            }
        };
        compare("row", d_temp, row_ref);
        compare("column", d_output, ref);
        verify_ok = errors == 0;
        fprintf(stderr, "%s: %zu mismatches in %zu values (%zu pixels per stage)\n",
                verify_ok ? "PASS" : "FAIL", errors, 2 * count, count);
    } catch (const std::exception& error) {
        fprintf(stderr, "FAIL: complete CPU reference unavailable: %s\n", error.what());
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
