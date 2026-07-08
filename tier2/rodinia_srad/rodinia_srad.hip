// rodinia_srad.hip — Rodinia SRAD benchmark (HIP, self-contained)
//
// Speckle-Reducing Anisotropic Diffusion (SRAD): iterative image smoothing
// using a Perona-Malik anisotropic diffusion scheme on a 2D float image.
//
// Two-phase stencil per iteration:
//   Phase 1 (srad1): Compute directional gradients (dN, dS, dW, dE) and the
//                    diffusion coefficient c for each pixel.
//   Phase 2 (srad2): Update the image J using the diffusion coefficients.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_srad [--image_size N] [--num_iterations K] [--block_size B]
//
//   --image_size N      Image dimension (N×N)              (default: 512)
//   --num_iterations K  SRAD iterations per pass           (default: 50)
//   --block_size B      Thread-block dimension (B×B)       (default: 16)
//
// Environment variables (override argv):
//   BENCH_PARAM_image_size, BENCH_PARAM_num_iterations, BENCH_PARAM_block_size
//   BENCH_PARAM_num_warmup   (warmup SRAD passes, default 3)
//   BENCH_PARAM_num_iters    (timed SRAD passes, each recorded, default 7)
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// kernel (srad1, srad2) carrying that kernel's accumulated GPU event time
// (time_ms) and host time (host_time_ms) across the K SRAD iterations within
// the pass, then a summary whose total_time_ms is the SUM of all recorded
// GPU event times. Effective bandwidth is derived downstream from time_ms
// and params.

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

// ---------------------------------------------------------------------------
// SRAD Kernels
// ---------------------------------------------------------------------------

// Phase 1: Compute directional gradients and diffusion coefficient c.
//
//   dN[i] = J[iN*cols+col] - J[idx]   (north gradient)
//   dS[i] = J[iS*cols+col] - J[idx]   (south gradient)
//   dW[i] = J[row*cols+jW] - J[idx]   (west gradient)
//   dE[i] = J[row*cols+jE] - J[idx]   (east gradient)
//
//   G2   = (dN² + dS² + dW² + dE²) / Jc²
//   L    = (dN + dS + dW + dE) / Jc
//   qsqr = (0.5*G2 - (1/16)*L²) / (1 + 0.25*L)²
//   c[i] = 1 / (1 + (qsqr - q0sqr) / (q0sqr * (1 + q0sqr)))   [clamped 0..1]
//
__global__ void srad1(const float* __restrict__ J,
                      float* __restrict__ dN, float* __restrict__ dS,
                      float* __restrict__ dW, float* __restrict__ dE,
                      float* __restrict__ c,
                      int rows, int cols, float q0sqr) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;

    // Clamped neighbor indices
    int iN = (row > 0)        ? (row - 1) : 0;
    int iS = (row < rows - 1) ? (row + 1) : (rows - 1);
    int jW = (col > 0)        ? (col - 1) : 0;
    int jE = (col < cols - 1) ? (col + 1) : (cols - 1);

    float Jc = J[idx];

    float dn = J[iN * cols + col] - Jc;
    float ds = J[iS * cols + col] - Jc;
    float dw = J[row * cols + jW] - Jc;
    float de = J[row * cols + jE] - Jc;

    dN[idx] = dn;
    dS[idx] = ds;
    dW[idx] = dw;
    dE[idx] = de;

    float G2   = (dn*dn + ds*ds + dw*dw + de*de) / (Jc * Jc);
    float L    = (dn + ds + dw + de) / Jc;
    float num  = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
    float den  = 1.0f + (0.25f * L);
    float qsqr = num / (den * den);

    // Perona-Malik diffusion coefficient
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    float ci = 1.0f / (1.0f + den);
    if (ci < 0.0f) ci = 0.0f;
    if (ci > 1.0f) ci = 1.0f;
    c[idx] = ci;
}

// Phase 2: Update image using the diffusion coefficients.
//
//   cN = c[idx]          cS = c[iS*cols+col]
//   cW = c[idx]          cE = c[row*cols+jE]
//   D  = cN*dN + cS*dS + cW*dW + cE*dE
//   J[idx] += 0.25 * lambda * D
//
__global__ void srad2(float* __restrict__ J,
                      const float* __restrict__ dN, const float* __restrict__ dS,
                      const float* __restrict__ dW, const float* __restrict__ dE,
                      const float* __restrict__ c,
                      int rows, int cols, float lambda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;

    // South and East neighbor diffusion coefficients
    int iS = (row < rows - 1) ? (row + 1) : (rows - 1);
    int jE = (col < cols - 1) ? (col + 1) : (cols - 1);

    float cN = c[idx];
    float cS = c[iS * cols + col];
    float cW = c[idx];
    float cE = c[row * cols + jE];

    float D = cN * dN[idx] + cS * dS[idx] + cW * dW[idx] + cE * dE[idx];

    J[idx] += 0.25f * lambda * D;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults
    int image_size  = 512;
    int num_iters   = 50;
    int block_dim   = 16;

    // Command-line fallback
    image_size = parseIntParam(argc, argv, "--image_size", image_size);
    num_iters  = parseIntParam(argc, argv, "--num_iterations", num_iters);
    block_dim  = parseIntParam(argc, argv, "--block_size", block_dim);

    float lambda = 0.25f;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--lambda") == 0) {
            lambda = (float)atof(argv[i + 1]);
            break;
        }
    }

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_image_size");
    if (env_val) image_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_iterations");
    if (env_val) num_iters = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_dim = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_bench_iters = bench_num_iters();

    int rows = image_size;
    int cols = image_size;
    int size = rows * cols;

    // Speckle noise parameter (variance of uniform noise)
    float q0sqr = 0.05f;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Image: %d×%d  |  SRAD iters/launch: %d\n\n",
            rows, cols, num_iters);

    // Host allocation — random float image in [0, 1]
    float* h_J      = (float*)malloc(size * sizeof(float));
    float* h_J_orig = (float*)malloc(size * sizeof(float));
    srand(42);
    for (int i = 0; i < size; ++i) {
        h_J[i] = (float)rand() / (float)RAND_MAX;
    }
    memcpy(h_J_orig, h_J, size * sizeof(float));

    // Device allocations
    float *d_J, *d_dN, *d_dS, *d_dW, *d_dE, *d_c;
    HIP_CHECK(hipMalloc(&d_J,  size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dN, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dS, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dW, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dE, size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c,  size * sizeof(float)));

    dim3 block(block_dim, block_dim);
    dim3 grid((cols + block_dim - 1) / block_dim,
              (rows + block_dim - 1) / block_dim);

    char params[128];
    snprintf(params, sizeof(params),
             "\"image_size\":%d,\"num_iterations\":%d,\"block_size\":%d",
             image_size, num_iters, block_dim);

    // Per-kernel accumulators: each pass launches srad1 and srad2 once per
    // SRAD iteration; the accumulated GPU/host time of all K iterations
    // within one pass becomes that pass's record for the kernel.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_srad1, acc_srad2;

    // One full SRAD pass. The diffusion updates the image J in place, so J is
    // reset to the pristine input at the start of every pass (untimed) —
    // each pass smooths the same image.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_srad1);
        bench_lap_reset(&acc_srad2);

        HIP_CHECK(hipMemcpy(d_J, h_J_orig, size * sizeof(float),
                            hipMemcpyHostToDevice));

        for (int k = 0; k < num_iters; ++k) {
            bench_lap_add(&acc_srad1, bench_time_op(&timer, [&]() {
                srad1<<<grid, block>>>(d_J, d_dN, d_dS, d_dW, d_dE, d_c, rows, cols, q0sqr);
            }));
            bench_lap_add(&acc_srad2, bench_time_op(&timer, [&]() {
                srad2<<<grid, block>>>(d_J, d_dN, d_dS, d_dW, d_dE, d_c, rows, cols, lambda);
            }));
        }
    };

    // Warmup passes (records not emitted)
    for (int w = 0; w < num_warmup; ++w) {
        run_pass();
    }

    // Timed passes — one record per kernel per pass
    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_bench_iters; ++i) {
        run_pass();
        bench_emit_iter("srad1", i, &acc_srad1, params);
        bench_emit_iter("srad2", i, &acc_srad2, params);
        total_gpu_ms += acc_srad1.gpu_ms + acc_srad2.gpu_ms;
        fprintf(stderr, "pass %d: srad1 gpu = %.3f ms, srad2 gpu = %.3f ms\n",
                i, acc_srad1.gpu_ms, acc_srad2.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d passes (%d SRAD iters/pass)\n",
            total_gpu_ms, num_bench_iters, num_iters);

    HIP_CHECK(hipFree(d_J));
    HIP_CHECK(hipFree(d_dN));
    HIP_CHECK(hipFree(d_dS));
    HIP_CHECK(hipFree(d_dW));
    HIP_CHECK(hipFree(d_dE));
    HIP_CHECK(hipFree(d_c));
    free(h_J);
    free(h_J_orig);

    return 0;
}
