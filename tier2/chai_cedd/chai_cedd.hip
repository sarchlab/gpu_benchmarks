// chai_cedd.hip — Canny Edge Detection benchmark (HIP)
//
// 5-stage pipeline: Gaussian blur, Sobel gradient, magnitude+direction,
// non-maximum suppression, hysteresis thresholding.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./chai_cedd [W=2048] [H=2048]
//
//   W=<width>    Image width  (default: 2048)
//   H=<height>   Image height (default: 2048)
//
// Environment variables (override argv):
//   BENCH_PARAM_width, BENCH_PARAM_height,
//   BENCH_PARAM_threshold_low, BENCH_PARAM_threshold_high
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// pipeline kernel carrying that kernel's GPU event time and host wall-clock
// time within the pass, then a summary whose total_time_ms is the SUM of all
// recorded GPU event times. Throughput (Mpixels/sec) is derived downstream
// from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing (key=value style)
// ---------------------------------------------------------------------------

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    size_t nlen = strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], name, nlen) == 0) {
            int v = atoi(argv[i] + nlen);
            if (v > 0) return v;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Kernel 1: Gaussian blur (5x5)
// ---------------------------------------------------------------------------

__constant__ float d_gauss[25] = {
    2.f/159, 4.f/159,  5.f/159,  4.f/159, 2.f/159,
    4.f/159, 9.f/159, 12.f/159,  9.f/159, 4.f/159,
    5.f/159,12.f/159, 15.f/159, 12.f/159, 5.f/159,
    4.f/159, 9.f/159, 12.f/159,  9.f/159, 4.f/159,
    2.f/159, 4.f/159,  5.f/159,  4.f/159, 2.f/159
};

__global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int nx = min(max(x + kx, 0), W - 1);
            int ny = min(max(y + ky, 0), H - 1);
            sum += input[ny * W + nx] * d_gauss[(ky + 2) * 5 + (kx + 2)];
        }
    }
    output[y * W + x] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: Sobel gradient (Gx, Gy)
// ---------------------------------------------------------------------------

__global__ void sobel_kernel(
    const float* __restrict__ blurred,
    float* __restrict__ gx,
    float* __restrict__ gy,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float sx = 0.0f, sy = 0.0f;

    // Sobel X kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
    // Sobel Y kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
    const int sobel_x[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int sobel_y[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int nx = min(max(x + kx, 0), W - 1);
            int ny = min(max(y + ky, 0), H - 1);
            float val = blurred[ny * W + nx];
            sx += val * sobel_x[ky + 1][kx + 1];
            sy += val * sobel_y[ky + 1][kx + 1];
        }
    }
    gx[y * W + x] = sx;
    gy[y * W + x] = sy;
}

// ---------------------------------------------------------------------------
// Kernel 3: Magnitude and direction
// ---------------------------------------------------------------------------

__global__ void magnitude_direction_kernel(
    const float* __restrict__ gx,
    const float* __restrict__ gy,
    float* __restrict__ mag,
    int* __restrict__ dir,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    float vx = gx[idx], vy = gy[idx];
    mag[idx] = sqrtf(vx * vx + vy * vy);

    // Quantize angle to 0, 45, 90, 135 degrees
    float angle = atan2f(vy, vx) * (180.0f / 3.14159265f);
    if (angle < 0) angle += 180.0f;

    int d;
    if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f))
        d = 0;   // horizontal
    else if (angle >= 22.5f && angle < 67.5f)
        d = 45;
    else if (angle >= 67.5f && angle < 112.5f)
        d = 90;  // vertical
    else
        d = 135;

    dir[idx] = d;
}

// ---------------------------------------------------------------------------
// Kernel 4: Non-maximum suppression
// ---------------------------------------------------------------------------

__global__ void nms_kernel(
    const float* __restrict__ mag,
    const int* __restrict__ dir,
    float* __restrict__ nms_out,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    float m = mag[idx];
    int d = dir[idx];

    float n1 = 0.0f, n2 = 0.0f;
    if (d == 0) {
        // Compare left/right
        if (x > 0) n1 = mag[y * W + (x - 1)];
        if (x < W - 1) n2 = mag[y * W + (x + 1)];
    } else if (d == 90) {
        // Compare up/down
        if (y > 0) n1 = mag[(y - 1) * W + x];
        if (y < H - 1) n2 = mag[(y + 1) * W + x];
    } else if (d == 45) {
        // Compare diagonal
        if (x < W - 1 && y > 0) n1 = mag[(y - 1) * W + (x + 1)];
        if (x > 0 && y < H - 1) n2 = mag[(y + 1) * W + (x - 1)];
    } else { // 135
        if (x > 0 && y > 0) n1 = mag[(y - 1) * W + (x - 1)];
        if (x < W - 1 && y < H - 1) n2 = mag[(y + 1) * W + (x + 1)];
    }

    nms_out[idx] = (m >= n1 && m >= n2) ? m : 0.0f;
}

// ---------------------------------------------------------------------------
// Kernel 5: Hysteresis thresholding
// ---------------------------------------------------------------------------

__global__ void hysteresis_kernel(
    const float* __restrict__ nms,
    unsigned char* __restrict__ edges,
    int W, int H,
    float low_thresh, float high_thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    float val = nms[idx];

    if (val >= high_thresh) {
        edges[idx] = 255;
    } else if (val >= low_thresh) {
        // Check if any 8-neighbor is a strong edge
        unsigned char is_edge = 0;
        for (int ky = -1; ky <= 1 && !is_edge; ++ky) {
            for (int kx = -1; kx <= 1 && !is_edge; ++kx) {
                if (kx == 0 && ky == 0) continue;
                int nx = x + kx, ny = y + ky;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    if (nms[ny * W + nx] >= high_thresh) {
                        is_edge = 255;
                    }
                }
            }
        }
        edges[idx] = is_edge;
    } else {
        edges[idx] = 0;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: simple Canny edge detection (for verification subset)
// ---------------------------------------------------------------------------

static void canny_cpu(const float* input, unsigned char* output, int W, int H,
                      float low_thresh, float high_thresh) {
    // Gaussian kernel
    const float gauss[25] = {
        2.f/159, 4.f/159,  5.f/159,  4.f/159, 2.f/159,
        4.f/159, 9.f/159, 12.f/159,  9.f/159, 4.f/159,
        5.f/159,12.f/159, 15.f/159, 12.f/159, 5.f/159,
        4.f/159, 9.f/159, 12.f/159,  9.f/159, 4.f/159,
        2.f/159, 4.f/159,  5.f/159,  4.f/159, 2.f/159
    };

    int npix = W * H;
    std::vector<float> blurred(npix), gx(npix), gy(npix), mag(npix), nms(npix);
    std::vector<int> dir(npix);

    // Stage 1: Gaussian blur
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int nx = std::min(std::max(x + kx, 0), W - 1);
                    int ny = std::min(std::max(y + ky, 0), H - 1);
                    sum += input[ny * W + nx] * gauss[(ky + 2) * 5 + (kx + 2)];
                }
            }
            blurred[y * W + x] = sum;
        }
    }

    // Stage 2: Sobel
    const int sobel_x[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int sobel_y[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sx = 0.0f, sy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = std::min(std::max(x + kx, 0), W - 1);
                    int ny = std::min(std::max(y + ky, 0), H - 1);
                    float val = blurred[ny * W + nx];
                    sx += val * sobel_x[ky + 1][kx + 1];
                    sy += val * sobel_y[ky + 1][kx + 1];
                }
            }
            gx[y * W + x] = sx;
            gy[y * W + x] = sy;
        }
    }

    // Stage 3: Magnitude + direction
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float vx = gx[idx], vy = gy[idx];
            mag[idx] = sqrtf(vx * vx + vy * vy);
            float angle = atan2f(vy, vx) * (180.0f / 3.14159265f);
            if (angle < 0) angle += 180.0f;
            if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f))
                dir[idx] = 0;
            else if (angle >= 22.5f && angle < 67.5f)
                dir[idx] = 45;
            else if (angle >= 67.5f && angle < 112.5f)
                dir[idx] = 90;
            else
                dir[idx] = 135;
        }
    }

    // Stage 4: NMS
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float m = mag[idx];
            int d = dir[idx];
            float n1 = 0.0f, n2 = 0.0f;
            if (d == 0) {
                if (x > 0) n1 = mag[y * W + (x - 1)];
                if (x < W - 1) n2 = mag[y * W + (x + 1)];
            } else if (d == 90) {
                if (y > 0) n1 = mag[(y - 1) * W + x];
                if (y < H - 1) n2 = mag[(y + 1) * W + x];
            } else if (d == 45) {
                if (x < W - 1 && y > 0) n1 = mag[(y - 1) * W + (x + 1)];
                if (x > 0 && y < H - 1) n2 = mag[(y + 1) * W + (x - 1)];
            } else {
                if (x > 0 && y > 0) n1 = mag[(y - 1) * W + (x - 1)];
                if (x < W - 1 && y < H - 1) n2 = mag[(y + 1) * W + (x + 1)];
            }
            nms[idx] = (m >= n1 && m >= n2) ? m : 0.0f;
        }
    }

    // Stage 5: Hysteresis
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float val = nms[idx];
            if (val >= high_thresh) {
                output[idx] = 255;
            } else if (val >= low_thresh) {
                unsigned char is_edge = 0;
                for (int ky = -1; ky <= 1 && !is_edge; ++ky) {
                    for (int kx = -1; kx <= 1 && !is_edge; ++kx) {
                        if (kx == 0 && ky == 0) continue;
                        int nx = x + kx, ny = y + ky;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            if (nms[ny * W + nx] >= high_thresh) {
                                is_edge = 255;
                            }
                        }
                    }
                }
                output[idx] = is_edge;
            } else {
                output[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int width          = 2048;
    int height         = 2048;
    int threshold_low  = 50;
    int threshold_high = 150;

    // Command-line fallback
    width  = parseIntParam(argc, argv, "W=", width);
    height = parseIntParam(argc, argv, "H=", height);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_width");
    if (env_val) width = atoi(env_val);
    env_val = getenv("BENCH_PARAM_height");
    if (env_val) height = atoi(env_val);
    env_val = getenv("BENCH_PARAM_threshold_low");
    if (env_val) threshold_low = atoi(env_val);
    env_val = getenv("BENCH_PARAM_threshold_high");
    if (env_val) threshold_high = atoi(env_val);

    int W = width;
    int H = height;
    int npix = W * H;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Canny Edge Detection  |  Image: %dx%d\n\n", W, H);

    // Generate synthetic grayscale image
    std::vector<float> h_input(npix);
    srand(42);
    for (int i = 0; i < npix; ++i) {
        h_input[i] = (float)(rand() % 256);
    }

    // Allocate device memory
    float *d_input, *d_blurred, *d_gx, *d_gy, *d_mag, *d_nms;
    int *d_dir;
    unsigned char *d_edges;

    HIP_CHECK(hipMalloc(&d_input,   npix * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_blurred, npix * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gx,      npix * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gy,      npix * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_mag,     npix * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dir,     npix * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_nms,     npix * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_edges,   npix * sizeof(unsigned char)));

    HIP_CHECK(hipMemcpy(d_input, h_input.data(), npix * sizeof(float), hipMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16);

    float low_t = (float)threshold_low, high_t = (float)threshold_high;

    char params[192];
    snprintf(params, sizeof(params),
             "\"width\":%d,\"height\":%d,\"threshold_low\":%d,\"threshold_high\":%d",
             width, height, threshold_low, threshold_high);

    // Per-kernel accumulators: each pass launches every pipeline kernel once;
    // the measured GPU/host time of each launch becomes that pass's record
    // for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_gauss, acc_sobel, acc_magdir, acc_nms, acc_hyst;

    // One full 5-stage CEDD pass. The pipeline is feed-forward: d_input is
    // never written and every intermediate buffer is fully overwritten each
    // pass, so no state reset is needed between passes.
    auto run_cedd_pass = [&]() {
        bench_lap_reset(&acc_gauss);
        bench_lap_reset(&acc_sobel);
        bench_lap_reset(&acc_magdir);
        bench_lap_reset(&acc_nms);
        bench_lap_reset(&acc_hyst);

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_gauss, bench_time_op(&timer, [&]() {
            gaussian_blur_kernel<<<gridDim, blockDim>>>(d_input, d_blurred, W, H);
        }));
        bench_lap_add(&acc_sobel, bench_time_op(&timer, [&]() {
            sobel_kernel<<<gridDim, blockDim>>>(d_blurred, d_gx, d_gy, W, H);
        }));
        bench_lap_add(&acc_magdir, bench_time_op(&timer, [&]() {
            magnitude_direction_kernel<<<gridDim, blockDim>>>(d_gx, d_gy, d_mag, d_dir, W, H);
        }));
        bench_lap_add(&acc_nms, bench_time_op(&timer, [&]() {
            nms_kernel<<<gridDim, blockDim>>>(d_mag, d_dir, d_nms, W, H);
        }));
        bench_lap_add(&acc_hyst, bench_time_op(&timer, [&]() {
            hysteresis_kernel<<<gridDim, blockDim>>>(d_nms, d_edges, W, H, low_t, high_t);
        }));
    };

    double total_gpu_ms = 0.0;
    run_cedd_pass();
    bench_emit_iter("gaussian_blur_kernel",       0, &acc_gauss,  params);
    bench_emit_iter("sobel_kernel",               0, &acc_sobel,  params);
    bench_emit_iter("magnitude_direction_kernel", 0, &acc_magdir, params);
    bench_emit_iter("nms_kernel",                 0, &acc_nms,    params);
    bench_emit_iter("hysteresis_kernel",          0, &acc_hyst,   params);
    total_gpu_ms += acc_gauss.gpu_ms + acc_sobel.gpu_ms + acc_magdir.gpu_ms +
                    acc_nms.gpu_ms + acc_hyst.gpu_ms;
    fprintf(stderr, "gauss=%.3f sobel=%.3f magdir=%.3f nms=%.3f hyst=%.3f ms\n",
            acc_gauss.gpu_ms, acc_sobel.gpu_ms, acc_magdir.gpu_ms,
            acc_nms.gpu_ms, acc_hyst.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification: compare a small sub-region with CPU reference
    // -------------------------------------------------------------------
    {
        // Verify a 128x128 sub-region at top-left
        int vW = (W < 128) ? W : 128;
        int vH = (H < 128) ? H : 128;

        // Build the sub-region input
        std::vector<float> sub_input(vW * vH);
        for (int y = 0; y < vH; ++y) {
            for (int x = 0; x < vW; ++x) {
                sub_input[y * vW + x] = h_input[y * W + x];
            }
        }

        std::vector<unsigned char> cpu_edges(vW * vH);
        canny_cpu(sub_input.data(), cpu_edges.data(), vW, vH, low_t, high_t);

        // Get GPU results for the same sub-region
        std::vector<unsigned char> h_edges(npix);
        HIP_CHECK(hipMemcpy(h_edges.data(), d_edges, npix * sizeof(unsigned char),
                             hipMemcpyDeviceToHost));

        // Note: GPU processes full image so border effects at (128,128) differ
        // from CPU processing only the sub-image. Compare interior only,
        // skipping a 6px border: the pipeline's influence radius is ~5px
        // (gaussian 2 + sobel 1 + nms 1 + hysteresis 1), so pixels closer to
        // the sub-region border can legitimately differ.
        int errors = 0;
        int compared = 0;
        for (int y = 6; y < vH - 6; ++y) {
            for (int x = 6; x < vW - 6; ++x) {
                unsigned char gpu_val = h_edges[y * W + x];
                unsigned char cpu_val = cpu_edges[y * vW + x];
                if (gpu_val != cpu_val) {
                    errors++;
                }
                compared++;
            }
        }

        float error_rate = compared > 0 ? (float)errors / compared * 100.0f : 0.0f;
        if (error_rate < 1.0f) {
            fprintf(stderr, "PASS (%.1f%% pixel mismatch in %dx%d interior, %d/%d)\n",
                    error_rate, vW - 12, vH - 12, errors, compared);
        } else {
            fprintf(stderr, "FAIL (%.1f%% pixel mismatch in %dx%d interior, %d/%d)\n",
                    error_rate, vW - 12, vH - 12, errors, compared);
            verify_ok = false;
        }
    }

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_blurred));
    HIP_CHECK(hipFree(d_gx));
    HIP_CHECK(hipFree(d_gy));
    HIP_CHECK(hipFree(d_mag));
    HIP_CHECK(hipFree(d_dir));
    HIP_CHECK(hipFree(d_nms));
    HIP_CHECK(hipFree(d_edges));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
