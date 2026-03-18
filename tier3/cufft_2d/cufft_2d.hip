/**
 * cufft_2d.hip — 2D complex single-precision FFT benchmark.
 *
 * Wraps cuFFT (NVIDIA) or hipFFT (AMD).
 *
 * Parameters:
 *   NX = NY = 2048
 *   Complex single-precision input: NX*NY cufftComplex / hipfftComplex values
 *   Forward C2C FFT (CUFFT_FORWARD / HIPFFT_FORWARD)
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations; report average
 *
 * Verification: 2D sinusoidal input at frequencies (kx=3, ky=5)
 *   → magnitude at bin [3][5] ≈ NX*NY/2
 *
 * GB/s = 2 * NX*NY * 2 * sizeof(float) / time_s / 1e9
 *        (read input + write output, each NX*NY complex floats)
 *
 * Output:
 *   cufft_2d,2048,<time_ms>,<GBs>
 *
 * Build:
 *   CUDA:  nvcc -O3 -std=c++14 -x cu -DUSE_CUFFT  cufft_2d.hip -lcufft  -o cufft_2d
 *   ROCm:  hipcc -O3 -std=c++14     -DUSE_HIPFFT cufft_2d.hip -lhipfft -o cufft_2d
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Platform-specific includes and macros
 * ---------------------------------------------------------------------- */
#ifdef USE_CUFFT
#  include <cufft.h>
#  include <cuda_runtime.h>

   typedef cufftComplex  fft_complex_t;
   typedef cufftHandle   fft_plan_t;

#  define GPU_CHECK(x) do {                                                  \
       cudaError_t _e = (x);                                                 \
       if (_e != cudaSuccess) {                                               \
           fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#  define FFT_CHECK(x) do {                                                  \
       cufftResult _e = (x);                                                 \
       if (_e != CUFFT_SUCCESS) {                                             \
           fprintf(stderr, "cuFFT error %s:%d: %d\n",                        \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

   typedef cudaEvent_t gpu_event_t;
#  define GPU_EVENT_CREATE(e)         GPU_CHECK(cudaEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)         GPU_CHECK(cudaEventRecord(e, 0))
#  define GPU_EVENT_SYNC(e)           GPU_CHECK(cudaEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a, b, ms) GPU_CHECK(cudaEventElapsedTime(&(ms), (a), (b)))
#  define GPU_DEVICE_SYNC()           GPU_CHECK(cudaDeviceSynchronize())
#  define GPU_MALLOC(p, n)            GPU_CHECK(cudaMalloc((p), (n)))
#  define GPU_FREE(p)                 GPU_CHECK(cudaFree(p))
#  define GPU_MEMCPY_H2D(d, h, n)    GPU_CHECK(cudaMemcpy((d), (h), (n), cudaMemcpyHostToDevice))
#  define GPU_MEMCPY_D2H(h, d, n)    GPU_CHECK(cudaMemcpy((h), (d), (n), cudaMemcpyDeviceToHost))

#  define FFT_PLAN_2D(plan, nx, ny)   FFT_CHECK(cufftPlan2d(&(plan), (nx), (ny), CUFFT_C2C))
#  define FFT_EXEC(plan, in, out)     FFT_CHECK(cufftExecC2C((plan), (in), (out), CUFFT_FORWARD))
#  define FFT_DESTROY(plan)           FFT_CHECK(cufftDestroy(plan))

#elif defined(USE_HIPFFT)
#  include <hipfft/hipfft.h>
#  include <hip/hip_runtime.h>

   typedef hipfftComplex fft_complex_t;
   typedef hipfftHandle  fft_plan_t;

#  define GPU_CHECK(x) do {                                                  \
       hipError_t _e = (x);                                                  \
       if (_e != hipSuccess) {                                                \
           fprintf(stderr, "HIP error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, hipGetErrorString(_e));                \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

#  define FFT_CHECK(x) do {                                                  \
       hipfftResult _e = (x);                                                \
       if (_e != HIPFFT_SUCCESS) {                                            \
           fprintf(stderr, "hipFFT error %s:%d: %d\n",                       \
                   __FILE__, __LINE__, (int)_e);                              \
           exit(1);                                                           \
       }                                                                      \
   } while (0)

   typedef hipEvent_t gpu_event_t;
#  define GPU_EVENT_CREATE(e)         GPU_CHECK(hipEventCreate(&(e)))
#  define GPU_EVENT_RECORD(e)         GPU_CHECK(hipEventRecord((e), 0))
#  define GPU_EVENT_SYNC(e)           GPU_CHECK(hipEventSynchronize(e))
#  define GPU_EVENT_ELAPSED(a, b, ms) GPU_CHECK(hipEventElapsedTime(&(ms), (a), (b)))
#  define GPU_DEVICE_SYNC()           GPU_CHECK(hipDeviceSynchronize())
#  define GPU_MALLOC(p, n)            GPU_CHECK(hipMalloc((p), (n)))
#  define GPU_FREE(p)                 GPU_CHECK(hipFree(p))
#  define GPU_MEMCPY_H2D(d, h, n)    GPU_CHECK(hipMemcpy((d), (h), (n), hipMemcpyHostToDevice))
#  define GPU_MEMCPY_D2H(h, d, n)    GPU_CHECK(hipMemcpy((h), (d), (n), hipMemcpyDeviceToHost))

#  define FFT_PLAN_2D(plan, nx, ny)   FFT_CHECK(hipfftPlan2d(&(plan), (nx), (ny), HIPFFT_C2C))
#  define FFT_EXEC(plan, in, out)     FFT_CHECK(hipfftExecC2C((plan), (in), (out), HIPFFT_FORWARD))
#  define FFT_DESTROY(plan)           FFT_CHECK(hipfftDestroy(plan))

#else
#  error "Define USE_CUFFT (NVIDIA) or USE_HIPFFT (AMD)"
#endif

/* -------------------------------------------------------------------------
 * Benchmark parameters
 * ---------------------------------------------------------------------- */
#define NX     2048
#define NY     2048
#define WARMUP 5
#define ITERS  5

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
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
    for (int i = 0; i < WARMUP; i++) {
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
    for (int i = 0; i < ITERS; i++) {
        FFT_EXEC(plan, d_in, d_out);
    }
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms / (float)ITERS;

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

    printf("cufft_2d,%d,%.3f,%.2f\n", NX, avg_ms, GBs);
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
