/**
 * cufft_1d_metal.mm — 1D complex FFT benchmark using Apple Accelerate vDSP.
 *
 * Uses vDSP_fft_zip (in-place complex FFT) from the Accelerate framework.
 * vDSP operates on split-complex format: separate arrays for real and imaginary parts.
 *
 * Parameters:
 *   N = 4194304 (1<<22)
 *   Complex single-precision input (split-complex format)
 *   Forward FFT (FFT_FORWARD)
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations; report average
 *
 * Verification: sinusoidal input at frequency 3 → magnitude at bin 3 ≈ N/2.
 * Note: vDSP_fft_zip computes the standard DFT (no extra scaling).
 * For sin(2π·3·n/N) input, the expected magnitude at bin 3 is N/2.
 *
 * GB/s = 2 * N * 2 * sizeof(float) / time_s / 1e9
 *
 * Output:
 *   cufft_1d,4194304,<time_ms>,<GBs>
 *
 * Build:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *     -framework Foundation -framework Accelerate \
 *     cufft_1d_metal.mm -o cufft_1d
 */

#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int N      = (1 << 22);   /* 4194304 complex samples */
static int WARMUP = 5;
static int ITERS  = 5;
static const char* DIRECTION = "forward";

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_warmup");
    if (env_val) WARMUP = atoi(env_val);
    env_val = getenv("BENCH_PARAM_iters");
    if (env_val) ITERS = atoi(env_val);
    env_val = getenv("BENCH_PARAM_direction");
    if (env_val) DIRECTION = env_val;

    /* ------------------------------------------------------------------ */
    /* Allocate split-complex buffers                                      */
    /* ------------------------------------------------------------------ */
    float *realp = (float *)malloc((size_t)N * sizeof(float));
    float *imagp = (float *)malloc((size_t)N * sizeof(float));

    if (!realp || !imagp) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Sinusoidal input at frequency k=3: x[n] = sin(2π·3·n/N) + 0·i
     * Expected after FFT: magnitude at bin 3 ≈ N/2 */
    for (int i = 0; i < N; i++) {
        realp[i] = sinf(2.0f * (float)M_PI * 3.0f * i / (float)N);
        imagp[i] = 0.0f;
    }

    DSPSplitComplex buf = { realp, imagp };

    /* ------------------------------------------------------------------ */
    /* Create FFT setup                                                    */
    /* ------------------------------------------------------------------ */
    const vDSP_Length log2N = (vDSP_Length)log2((double)N);
    FFTSetup setup = vDSP_create_fftsetup(log2N, FFT_RADIX2);
    if (!setup) {
        fprintf(stderr, "vDSP_create_fftsetup failed\n");
        free(realp);
        free(imagp);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Get timebase info for mach_absolute_time → nanoseconds              */
    /* ------------------------------------------------------------------ */
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    /* ------------------------------------------------------------------ */
    /* Warmup (5 iterations, not timed)                                   */
    /* ------------------------------------------------------------------ */
    for (int i = 0; i < WARMUP; i++) {
        vDSP_fft_zip(setup, &buf, 1, log2N, FFT_FORWARD);
    }

    /* Re-initialize input before timed runs.
     * (vDSP_fft_zip is in-place; warmup overwrites the buffer.)          */
    for (int i = 0; i < N; i++) {
        realp[i] = sinf(2.0f * (float)M_PI * 3.0f * i / (float)N);
        imagp[i] = 0.0f;
    }

    /* ------------------------------------------------------------------ */
    /* Timed iterations (time only the FFT calls, not data init)          */
    /* ------------------------------------------------------------------ */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < ITERS; iter++) {
        vDSP_fft_zip(setup, &buf, 1, log2N, FFT_FORWARD);
    }
    uint64_t t1 = mach_absolute_time();

    double total_ns = (double)(t1 - t0) * tb.numer / tb.denom;
    double avg_ns   = total_ns / ITERS;
    double avg_ms   = avg_ns / 1e6;

    /* ------------------------------------------------------------------ */
    /* Verification: run one fresh FFT on a clean sinusoidal input.       */
    /* The timed loop applied ITERS consecutive FFTs (in-place), so       */
    /* we cannot use that result for verification.                         */
    /* ------------------------------------------------------------------ */
    {
        float *vr = (float *)malloc((size_t)N * sizeof(float));
        float *vi = (float *)malloc((size_t)N * sizeof(float));
        if (!vr || !vi) {
            fprintf(stderr, "malloc for verification failed\n");
        } else {
            for (int i = 0; i < N; i++) {
                vr[i] = sinf(2.0f * (float)M_PI * 3.0f * i / (float)N);
                vi[i] = 0.0f;
            }
            DSPSplitComplex vbuf = { vr, vi };
            vDSP_fft_zip(setup, &vbuf, 1, log2N, FFT_FORWARD);

            float mag3     = sqrtf(vr[3] * vr[3] + vi[3] * vi[3]);
            float expected = (float)N / 2.0f;
            float rel_err  = fabsf(mag3 - expected) / expected;

            fprintf(stderr, "Verification: bin3 mag=%.1f  expected=%.1f  rel_err=%.4f  %s\n",
                    mag3, expected, rel_err, rel_err < 0.01f ? "PASS" : "FAIL");

            fprintf(stderr, "First 16 bin magnitudes:\n");
            for (int k = 0; k < 16; k++) {
                float mag = sqrtf(vr[k] * vr[k] + vi[k] * vi[k]);
                fprintf(stderr, "  bin[%2d] = %.2f\n", k, mag);
            }

            free(vr);
            free(vi);

            if (rel_err >= 0.01f) {
                fprintf(stderr, "Verification FAILED\n");
                vDSP_destroy_fftsetup(setup);
                free(realp);
                free(imagp);
                return 1;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Report                                                              */
    /* ------------------------------------------------------------------ */
    double bytes = 2.0 * N * 2 * sizeof(float);
    double GBs   = bytes / (avg_ms / 1000.0) / 1e9;

    /* JSON-lines output */
    printf("{\"type\":\"kernel\",\"name\":\"cufftExecC2C\",\"time_ms\":%.6f,"
           "\"params\":{\"n\":%d,\"warmup\":%d,\"iters\":%d,\"direction\":\"%s\"}}\n",
           avg_ms, N, WARMUP, ITERS, DIRECTION);
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f}]}\n",
           (double)(avg_ms * ITERS), GBs);
    fprintf(stderr, "cufft_1d  N=%d  time_ms=%.3f  GB/s=%.2f  (Accelerate vDSP_fft_zip)\n",
            N, avg_ms, GBs);

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                             */
    /* ------------------------------------------------------------------ */
    vDSP_destroy_fftsetup(setup);
    free(realp);
    free(imagp);
    return 0;
}
