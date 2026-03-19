// cuda_convolution_separable.hip — Separable 2D convolution benchmark (HIP)
//
// Two-pass separable 2D convolution: row pass + column pass.
// Measures throughput in GB/s = 2 * W * H * sizeof(float) / time_s / 1e9.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./cuda_convolution_separable [--width W] [--height H] [--radius R] [--iterations I]
//
//   --width W         Image width  (default: 4096)
//   --height H        Image height (default: 4096)
//   --radius R        Kernel radius (default: 8, filter width = 2*R+1)
//   --iterations I    Timed iterations (default: 5)
//
// Output (stdout): CSV row — cuda_convolution_separable,<WxH>,<time_ms>,<GBs>
// Output (stderr): human-readable results

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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
#define COL_TILE_H 128

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

static void conv_separable_cpu(const float* input, float* output,
                               int width, int height,
                               const float* kernel_weights, int radius) {
    int klen = 2 * radius + 1;

    // Temporary buffer for row pass
    std::vector<float> temp((size_t)width * height, 0.0f);

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
    int width  = parseIntParam(argc, argv, "--width",  4096);
    int height = parseIntParam(argc, argv, "--height", 4096);
    int radius = parseIntParam(argc, argv, "--radius", 8);
    int iters  = parseIntParam(argc, argv, "--iterations", 5);

    if (radius > MAX_KERNEL_RADIUS) {
        fprintf(stderr, "Error: radius=%d exceeds max=%d\n", radius, MAX_KERNEL_RADIUS);
        return 1;
    }

    int klen = 2 * radius + 1;
    size_t imageBytes = (size_t)width * height * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Image: %dx%d  |  Kernel radius: %d (%d-tap)  |  "
                    "Iterations: 5 warmup + %d timed\n\n",
            width, height, radius, klen, iters);

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

    // Copy kernel to constant memory
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_kernel), h_kernel.data(),
                                 klen * sizeof(float)));

    // Host allocation + initialization
    float* h_input  = (float*)malloc(imageBytes);
    float* h_output = (float*)malloc(imageBytes);
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = (float)(i % 256) / 255.0f;
    }

    // Device allocation
    float *d_input, *d_temp, *d_output;
    HIP_CHECK(hipMalloc(&d_input,  imageBytes));
    HIP_CHECK(hipMalloc(&d_temp,   imageBytes));
    HIP_CHECK(hipMalloc(&d_output, imageBytes));

    HIP_CHECK(hipMemcpy(d_input, h_input, imageBytes, hipMemcpyHostToDevice));

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

    // Lambda: run full separable convolution
    auto run_conv = [&]() {
        hipLaunchKernelGGL(convolution_row_kernel,
                           rowGrid, rowBlock, rowSmem, 0,
                           d_input, d_temp, width, height, radius);
        hipLaunchKernelGGL(convolution_col_kernel,
                           colGrid, colBlock, colSmem, 0,
                           d_temp, d_output, width, height, radius);
    };

    // -------------------------------------------------------------------
    // Warmup (5 iterations)
    // -------------------------------------------------------------------
    for (int w = 0; w < 5; ++w) {
        run_conv();
        HIP_CHECK(hipDeviceSynchronize());
    }

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    hipEvent_t evStart, evStop;
    HIP_CHECK(hipEventCreate(&evStart));
    HIP_CHECK(hipEventCreate(&evStop));

    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(evStart, 0));
        run_conv();
        HIP_CHECK(hipEventRecord(evStop, 0));
        HIP_CHECK(hipEventSynchronize(evStop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, evStart, evStop));
        times[i] = (double)ms;
    }

    HIP_CHECK(hipEventDestroy(evStart));
    HIP_CHECK(hipEventDestroy(evStop));

    // Compute average time
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) sum += times[i];
    double avg_ms = sum / iters;

    // GB/s = 2 * W * H * sizeof(float) / time_s / 1e9
    double data_bytes = 2.0 * (double)width * (double)height * sizeof(float);
    double gbs = data_bytes / (avg_ms * 1e-3) / 1e9;

    // CSV output to stdout
    printf("cuda_convolution_separable,%dx%d,%.4f,%.4f\n",
           width, height, avg_ms, gbs);

    // Human-readable output to stderr
    fprintf(stderr, "Average time: %.4f ms\n", avg_ms);
    fprintf(stderr, "Throughput:   %.4f GB/s\n", gbs);

    // -------------------------------------------------------------------
    // Verification (for smaller images)
    // -------------------------------------------------------------------
    int verify_w = (width  <= 512) ? width  : 512;
    int verify_h = (height <= 512) ? height : 512;

    {
        HIP_CHECK(hipMemcpy(h_output, d_output, imageBytes, hipMemcpyDeviceToHost));

        // CPU reference on the full image
        std::vector<float> ref((size_t)width * height);
        conv_separable_cpu(h_input, ref.data(), width, height,
                           h_kernel.data(), radius);

        int errors = 0;
        for (int r = 0; r < verify_h; ++r) {
            for (int c = 0; c < verify_w; ++c) {
                int idx = r * width + c;
                float diff = fabsf(h_output[idx] - ref[idx]);
                float tol = 1e-3f * fabsf(ref[idx]) + 1e-5f;
                if (diff > tol) {
                    if (errors < 10) {
                        fprintf(stderr, "Mismatch at (%d,%d): GPU=%.6f CPU=%.6f diff=%.6f\n",
                                r, c, h_output[idx], ref[idx], diff);
                    }
                    errors++;
                }
            }
        }

        if (errors > 0)
            fprintf(stderr, "FAIL: %d errors in %dx%d verification region\n",
                    errors, verify_w, verify_h);
        else
            fprintf(stderr, "PASS\n");
    }

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
