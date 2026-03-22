/**
 * cufft_2d.cu — 2D complex single-precision FFT benchmark.
 *
 * Wraps cuFFT (NVIDIA).
 *
 * Parameters (from env vars BENCH_PARAM_*):
 *   grid_dim  — square grid dimension NX=NY for the 2D FFT
 *   warmup    — warmup iterations (not timed)
 *   iters     — timed iterations
 *   direction — FFT direction (forward/inverse/roundtrip)
 *
 * Verification: 2D sinusoidal input at frequencies (kx=3, ky=5)
 *   → magnitude at bin [3][5] ≈ NX*NY/2
 *
 * GB/s = 2 * NX*NY * 2 * sizeof(float) / time_s / 1e9
 *        (read input + write output, each NX*NY complex floats)
 *
 * Output (stdout): JSON-lines protocol
 * Output (stderr): human-readable diagnostics
 *
 * Build:
 *   CUDA:  nvcc -O3 -std=c++14 cufft_2d.cu -lcufft  -o cufft_2d
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Platform-specific includes and macros
 * ---------------------------------------------------------------------- */
#include <cufft.h>
#include <cuda_runtime.h>

   typedef cufftComplex  fft_complex_t;
   typedef cufftHandle   fft_plan_t;

#define GPU_CHECK(x) do {                                                  \
       cudaError_t _e = (x);                                                 \
       if (_e != cudaSuccess) {                                               \
           fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#define FFT_CHECK(x) do {                                                  \
       cufftResult _e = (x);                                                 \
       if (_e != CUFFT_SUCCESS) {                                             \
           fprintf(stderr, "cuFFT error %s:%d: %d\n",                        \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

   typedef cudaEvent_t gpu_event_t;
#define GPU_EVENT_CREATE(e)         GPU_CHECK(cudaEventCreate(&(e)))
#define GPU_EVENT_RECORD(e)         GPU_CHECK(cudaEventRecord(e, 0))
#define GPU_EVENT_SYNC(e)           GPU_CHECK(cudaEventSynchronize(e))
#define GPU_EVENT_ELAPSED(a, b, ms) GPU_CHECK(cudaEventElapsedTime(&(ms), (a), (b)))
#define GPU_DEVICE_SYNC()           GPU_CHECK(cudaDeviceSynchronize())
#define GPU_MALLOC(p, n)            GPU_CHECK(cudaMalloc((p), (n)))
#define GPU_FREE(p)                 GPU_CHECK(cudaFree(p))
#define GPU_MEMCPY_H2D(d, h, n)    GPU_CHECK(cudaMemcpy((d), (h), (n), cudaMemcpyHostToDevice))
#define GPU_MEMCPY_D2H(h, d, n)    GPU_CHECK(cudaMemcpy((h), (d), (n), cudaMemcpyDeviceToHost))

#define FFT_PLAN_2D(plan, nx, ny)   FFT_CHECK(cufftPlan2d(&(plan), (nx), (ny), CUFFT_C2C))
#define FFT_EXEC(plan, in, out)     FFT_CHECK(cufftExecC2C((plan), (in), (out), CUFFT_FORWARD))
#define FFT_DESTROY(plan)           FFT_CHECK(cufftDestroy(plan))
#define KERNEL_NAME "cufftExecC2C"


/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Default parameters (overridden by BENCH_PARAM_* env vars)          */
    /* ------------------------------------------------------------------ */
    int grid_dim = 2048;
    int warmup   = 5;
    int iters    = 5;
    const char *direction = "forward";

    const char *env_val;
    env_val = getenv("BENCH_PARAM_grid_dim");
    if (env_val) grid_dim = atoi(env_val);
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) warmup = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) iters = atoi(env_val);
    env_val = getenv("BENCH_PARAM_direction");
    if (env_val) direction = env_val;

    int NX = grid_dim;
    int NY = grid_dim;
    long long total = (long long)NX * NY;

    /* ------------------------------------------------------------------ */
    /* Host allocation and initialization                                  */
    /* ------------------------------------------------------------------ */
    fft_complex_t *h_in  = (fft_complex_t *)malloc((size_t)total * sizeof(fft_complex_t));
    fft_complex_t *h_out = (fft_complex_t *)malloc((size_t)total * sizeof(fft_complex_t));

    if (!h_in || !h_out) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    /* 2D sinusoidal input at spatial frequencies (kx=3, ky=5):
     *   x[i][j].real = sin(2π·(3·i/NX + 5·j/NY)),   x[i][j].imag = 0
     *
     * After 2D forward FFT:
     *   |X[3][5]|  ≈ NX*NY / 2  (by separability and DFT of sin)
     *   |X[NX-3][NY-5]| ≈ NX*NY / 2
     */
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            h_in[i * NY + j].x = sinf(2.0f * (float)M_PI *
                                       (3.0f * i / (float)NX +
                                        5.0f * j / (float)NY));
            h_in[i * NY + j].y = 0.0f;
        }
    }
    memset(h_out, 0, (size_t)total * sizeof(fft_complex_t));

    /* ------------------------------------------------------------------ */
    /* Device allocation and upload                                        */
    /* ------------------------------------------------------------------ */
    fft_complex_t *d_in, *d_out;
    GPU_MALLOC((void **)&d_in,  (size_t)total * sizeof(fft_complex_t));
    GPU_MALLOC((void **)&d_out, (size_t)total * sizeof(fft_complex_t));
    GPU_MEMCPY_H2D(d_in, h_in, (size_t)total * sizeof(fft_complex_t));

    /* ------------------------------------------------------------------ */
    /* Create 2D FFT plan                                                  */
    /* ------------------------------------------------------------------ */
    fft_plan_t plan;
    FFT_PLAN_2D(plan, NX, NY);

    /* ------------------------------------------------------------------ */
    /* Warmup                                                              */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < warmup; i++) {
        FFT_EXEC(plan, d_in, d_out);
    }
    GPU_DEVICE_SYNC();

    /* ------------------------------------------------------------------ */
    /* Timed iterations                                                    */
    /* ------------------------------------------------------------------ */
    gpu_event_t ev_start, ev_stop;
    GPU_EVENT_CREATE(ev_start);
    GPU_EVENT_CREATE(ev_stop);

    GPU_EVENT_RECORD(ev_start);
    for (int i = 0; i < iters; i++) {
        FFT_EXEC(plan, d_in, d_out);
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)iters;

    /* ------------------------------------------------------------------ */
    /* Copy result back for verification                                   */
    /* ------------------------------------------------------------------ */
    GPU_MEMCPY_D2H(h_out, d_out, (size_t)total * sizeof(fft_complex_t));

    /* ------------------------------------------------------------------ */
    /* Verify: 2D sinusoidal at (kx=3, ky=5) → |X[3][5]| ≈ NX*NY/2      */
    /* ------------------------------------------------------------------ */
    {
        int idx = 3 * NY + 5;
        float mag35   = sqrtf(h_out[idx].x * h_out[idx].x +
                               h_out[idx].y * h_out[idx].y);
        float expected = (float)total / 2.0f;
        float rel_err  = fabsf(mag35 - expected) / expected;

        fprintf(stderr,
                "Verification: bin[3][5] mag=%.1f  expected=%.1f  rel_err=%.4f  %s\n",
                mag35, expected, rel_err, rel_err < 0.01f ? "PASS" : "FAIL");

        /* Show magnitudes of first 16 bins of row 0 and bin [3][5] */
        fprintf(stderr, "Row-0 bin magnitudes [0..15]:\n");
        for (int k = 0; k < 16; k++) {
            float mag = sqrtf(h_out[k].x * h_out[k].x +
                              h_out[k].y * h_out[k].y);
            fprintf(stderr, "  [0][%2d] = %.2f\n", k, mag);
        }

        if (rel_err >= 0.01f) {
            fprintf(stderr, "Verification FAILED\n");
            FFT_DESTROY(plan);
            GPU_FREE(d_in);
            GPU_FREE(d_out);
            free(h_in);
            free(h_out);
            return 1;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    /* GB/s: read NX*NY complex floats + write NX*NY complex floats
     * = 2 * NX*NY * sizeof(cufftComplex) / time_s / 1e9
     * = 2 * NX*NY * 2 * sizeof(float)   / time_s / 1e9            */
    double bytes = 2.0 * total * 2 * sizeof(float);
    double GBs   = bytes / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"grid_dim\":%d,\"warmup\":%d,\"iters\":%d,\"direction\":\"%s\"}}\n",
           KERNEL_NAME, (double)avg_ms, grid_dim, warmup, iters, direction);

    /* JSON-lines: summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f}]}\n",
           (double)total_ms, GBs);

    fprintf(stderr, "cufft_2d  NX=%d NY=%d  time_ms=%.3f  GB/s=%.2f\n",
            NX, NY, avg_ms, GBs);

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                             */
    /* ------------------------------------------------------------------ */
    FFT_DESTROY(plan);
    GPU_FREE(d_in);
    GPU_FREE(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
