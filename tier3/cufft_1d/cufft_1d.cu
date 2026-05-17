/**
 * cufft_1d.cu — 1D complex single-precision FFT benchmark.
 *
 * Wraps cuFFT (NVIDIA).
 *
 * Parameters:
 *   N = 4194304 (1<<22)
 *   Complex single-precision input: N cufftComplex values
 *   Forward C2C FFT (CUFFT_FORWARD)
 *
 * Warmup: configurable iterations (not timed)
 * Timed:  single iteration (repetition handled by CLI)
 *
 * Verification: sinusoidal input at frequency 3 → magnitude at bin 3 ≈ N/2
 *
 * GB/s = 2 * N * 2 * sizeof(float) / time_s / 1e9
 *        (read input + write output, each N complex floats)
 *
 * Output:
 *   cufft_1d,4194304,<time_ms>,<GBs>
 *
 * Build:
 *   CUDA:  nvcc -O3 -std=c++14 cufft_1d.cu -lcufft  -o cufft_1d
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

#define FFT_PLAN_1D(plan, n)        FFT_CHECK(cufftPlan1d(&(plan), (n), CUFFT_C2C, 1))
#define FFT_EXEC(plan, in, out)     FFT_CHECK(cufftExecC2C((plan), (in), (out), CUFFT_FORWARD))
#define FFT_DESTROY(plan)           FFT_CHECK(cufftDestroy(plan))


/* -------------------------------------------------------------------------
 * Benchmark parameters (mutable — overridable via BENCH_PARAM_* env vars)
 * ---------------------------------------------------------------------- */
static int N      = (1 << 22);   /* 4194304 complex samples */
static int WARMUP = 0;
static const char* DIRECTION = "forward";

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_direction");
    if (env_val) DIRECTION = env_val;

    /* ------------------------------------------------------------------ */
    /* Host allocation and initialization                                  */
    /* ------------------------------------------------------------------ */
    fft_complex_t *h_in  = (fft_complex_t *)malloc((size_t)N * sizeof(fft_complex_t));
    fft_complex_t *h_out = (fft_complex_t *)malloc((size_t)N * sizeof(fft_complex_t));

    if (!h_in || !h_out) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    /* Sinusoidal input at frequency k=3: x[n] = sin(2π·3·n/N) + 0·i
     * Expected: |X[3]| ≈ N/2,  |X[N-3]| ≈ N/2 */
    for (int i = 0; i < N; i++) {
        h_in[i].x = sinf(2.0f * (float)M_PI * 3.0f * i / (float)N);
        h_in[i].y = 0.0f;
    }
    memset(h_out, 0, (size_t)N * sizeof(fft_complex_t));

    /* ------------------------------------------------------------------ */
    /* Device allocation and upload                                        */
    /* ------------------------------------------------------------------ */
    fft_complex_t *d_in, *d_out;
    GPU_MALLOC((void **)&d_in,  (size_t)N * sizeof(fft_complex_t));
    GPU_MALLOC((void **)&d_out, (size_t)N * sizeof(fft_complex_t));
    GPU_MEMCPY_H2D(d_in, h_in, (size_t)N * sizeof(fft_complex_t));

    /* ------------------------------------------------------------------ */
    /* Create FFT plan                                                     */
    /* ------------------------------------------------------------------ */
    fft_plan_t plan;
    FFT_PLAN_1D(plan, N);

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
    FFT_EXEC(plan, d_in, d_out);
    GPU_EVENT_RECORD(ev_stop);
    GPU_EVENT_SYNC(ev_stop);

    float total_ms = 0.0f;
    GPU_EVENT_ELAPSED(ev_start, ev_stop, total_ms);
    float avg_ms = total_ms;

    /* ------------------------------------------------------------------ */
    /* Copy result back for verification                                   */
    /* ------------------------------------------------------------------ */
    GPU_MEMCPY_D2H(h_out, d_out, (size_t)N * sizeof(fft_complex_t));

    /* ------------------------------------------------------------------ */
    /* Verify: sinusoidal at freq 3 → |X[3]| ≈ N/2                        */
    /* ------------------------------------------------------------------ */
    {
        float mag3 = sqrtf(h_out[3].x * h_out[3].x + h_out[3].y * h_out[3].y);
        float expected = (float)N / 2.0f;
        float rel_err  = fabsf(mag3 - expected) / expected;
        fprintf(stderr, "Verification: bin3 mag=%.1f  expected=%.1f  rel_err=%.4f  %s\n",
                mag3, expected, rel_err, rel_err < 0.01f ? "PASS" : "FAIL");

        /* Also show first 16 bin magnitudes */
        fprintf(stderr, "First 16 bin magnitudes:\n");
        for (int k = 0; k < 16; k++) {
            float mag = sqrtf(h_out[k].x * h_out[k].x + h_out[k].y * h_out[k].y);
            fprintf(stderr, "  bin[%2d] = %.2f\n", k, mag);
        }

        if (rel_err >= 0.01f) {
            fprintf(stderr, "Verification FAILED\n");
            /* cleanup and exit with failure */
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
    /* GB/s: read N complex floats + write N complex floats
     * = 2 * N * sizeof(cufftComplex) / time_s / 1e9
     * = 2 * N * 2 * sizeof(float) / time_s / 1e9            */
    double bytes  = 2.0 * N * 2 * sizeof(float);
    double GBs    = bytes / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines output */
    printf("{\"type\":\"kernel\",\"name\":\"cufftExecC2C\",\"time_ms\":%.6f,"
           "\"params\":{\"n\":%d,\"warmup\":%d,\"direction\":\"%s\"}}\n",
           (double)avg_ms, N, WARMUP, DIRECTION);
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f}]}\n",
           (double)total_ms, GBs);
    fprintf(stderr, "cufft_1d  N=%d  time_ms=%.3f  GB/s=%.2f\n", N, avg_ms, GBs);

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
