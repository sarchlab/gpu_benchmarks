/**
 * cufft_2d_metal.mm — 2D complex FFT benchmark using Apple Accelerate vDSP.
 *
 * Performs a 2D forward complex FFT via two passes of 1D FFTs:
 *   1. Apply vDSP_fft_zip to each row  (NY-point FFT, stride=1)
 *   2. Apply vDSP_fft_zip to each column (NX-point FFT, via temp buffer)
 *
 * This is mathematically equivalent to a 2D DFT by the separability property.
 *
 * Parameters:
 *   NX = NY = 2048
 *   Complex single-precision input (split-complex format)
 *   Forward FFT (FFT_FORWARD)
 *
 * Warmup: 5 iterations (not timed)
 * Timed:  5 iterations; report average
 *
 * Verification: 2D sinusoidal input at (kx=3, ky=5)
 *   → magnitude at split-complex bin [3][5] ≈ NX*NY/2
 *
 * GB/s = 2 * NX*NY * 2 * sizeof(float) / time_s / 1e9
 *
 * Output:
 *   cufft_2d,2048,<time_ms>,<GBs>
 *
 * Build:
 *   clang++ -O2 -std=c++17 -fobjc-arc \
 *     -framework Foundation -framework Accelerate \
 *     cufft_2d_metal.mm -o cufft_2d
 */

#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX     2048
#define NY     2048
#define WARMUP 5
#define ITERS  5

/* -------------------------------------------------------------------------
 * fft_rows: apply a 1D forward FFT to each row of the split-complex matrix.
 *
 *   realp[i*NY .. i*NY+NY-1], imagp[i*NY .. i*NY+NY-1]  for i in [0, NX)
 *
 * vDSP_fft_zip operates in-place with unit stride, length 2^log2ny.
 * ---------------------------------------------------------------------- */
static void fft_rows(FFTSetup setup,
                     float *realp, float *imagp,
                     int nx, int ny,
                     vDSP_Length log2ny)
{
    for (int i = 0; i < nx; i++) {
        DSPSplitComplex row = { realp + (size_t)i * ny,
                                imagp + (size_t)i * ny };
        vDSP_fft_zip(setup, &row, 1, log2ny, FFT_FORWARD);
    }
}

/* -------------------------------------------------------------------------
 * fft_cols: apply a 1D forward FFT to each column of the split-complex matrix.
 *
 * Columns are not contiguous in memory, so we copy each column into a
 * temporary contiguous buffer, transform it, then scatter back.
 * ---------------------------------------------------------------------- */
static void fft_cols(FFTSetup setup,
                     float *realp, float *imagp,
                     int nx, int ny,
                     vDSP_Length log2nx)
{
    float *tmp_r = (float *)malloc((size_t)nx * sizeof(float));
    float *tmp_i = (float *)malloc((size_t)nx * sizeof(float));
    if (!tmp_r || !tmp_i) {
        fprintf(stderr, "fft_cols: malloc failed\n");
        exit(1);
    }

    for (int j = 0; j < ny; j++) {
        /* Gather column j */
        for (int i = 0; i < nx; i++) {
            tmp_r[i] = realp[(size_t)i * ny + j];
            tmp_i[i] = imagp[(size_t)i * ny + j];
        }
        /* Transform */
        DSPSplitComplex col = { tmp_r, tmp_i };
        vDSP_fft_zip(setup, &col, 1, log2nx, FFT_FORWARD);
        /* Scatter back */
        for (int i = 0; i < nx; i++) {
            realp[(size_t)i * ny + j] = tmp_r[i];
            imagp[(size_t)i * ny + j] = tmp_i[i];
        }
    }

    free(tmp_r);
    free(tmp_i);
}

/* -------------------------------------------------------------------------
 * fill_input: populate the split-complex matrix with the 2D sinusoidal
 *   x[i][j] = sin(2π·(3·i/NX + 5·j/NY)),  y[i][j] = 0
 * ---------------------------------------------------------------------- */
static void fill_input(float *realp, float *imagp, int nx, int ny)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            realp[(size_t)i * ny + j] =
                sinf(2.0f * (float)M_PI *
                     (3.0f * i / (float)nx + 5.0f * j / (float)ny));
            imagp[(size_t)i * ny + j] = 0.0f;
        }
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
    const vDSP_Length log2NX = 11;   /* log2(2048) = 11 */
    const vDSP_Length log2NY = 11;

    /* Create FFT setup (supports up to length 2^11 = 2048) */
    FFTSetup setup = vDSP_create_fftsetup(11, FFT_RADIX2);
    if (!setup) {
        fprintf(stderr, "vDSP_create_fftsetup failed\n");
        return 1;
    }

    long long total = (long long)NX * NY;

    /* Allocate split-complex buffers */
    float *realp = (float *)malloc((size_t)total * sizeof(float));
    float *imagp = (float *)malloc((size_t)total * sizeof(float));
    if (!realp || !imagp) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Get mach_absolute_time timebase */
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    /* ------------------------------------------------------------------ */
    /* Warmup (5 iterations, not timed)                                   */
    /* ------------------------------------------------------------------ */
    for (int w = 0; w < WARMUP; w++) {
        fill_input(realp, imagp, NX, NY);
        fft_rows(setup, realp, imagp, NX, NY, log2NY);
        fft_cols(setup, realp, imagp, NX, NY, log2NX);
    }

    /* Re-initialize before timed runs */
    fill_input(realp, imagp, NX, NY);

    /* ------------------------------------------------------------------ */
    /* Timed iterations (time only the FFT calls, not data init)          */
    /* ------------------------------------------------------------------ */
    uint64_t t0 = mach_absolute_time();
    for (int iter = 0; iter < ITERS; iter++) {
        fft_rows(setup, realp, imagp, NX, NY, log2NY);
        fft_cols(setup, realp, imagp, NX, NY, log2NX);
    }
    uint64_t t1 = mach_absolute_time();

    double total_ns = (double)(t1 - t0) * tb.numer / tb.denom;
    double avg_ns   = total_ns / ITERS;
    double avg_ms   = avg_ns / 1e6;

    /* ------------------------------------------------------------------ */
    /* Verification: run one fresh 2D FFT on a clean sinusoidal input.    */
    /* The timed loop applied ITERS consecutive in-place FFTs so we       */
    /* cannot use that result directly.                                    */
    /*                                                                     */
    /* Expected: |X[3][5]| = NX*NY/2 = 2097152                           */
    /* Derivation (by separability):                                       */
    /*   FFT of sin(2π(3i/NX + 5j/NY)) →  X[3][5] = -j·NX·NY/2          */
    /*   |X[3][5]| = NX·NY/2                                               */
    /* ------------------------------------------------------------------ */
    {
        float *vr = (float *)malloc((size_t)total * sizeof(float));
        float *vi = (float *)malloc((size_t)total * sizeof(float));
        if (!vr || !vi) {
            fprintf(stderr, "malloc for verification failed\n");
        } else {
            fill_input(vr, vi, NX, NY);
            fft_rows(setup, vr, vi, NX, NY, log2NY);
            fft_cols(setup, vr, vi, NX, NY, log2NX);

            int idx = 3 * NY + 5;
            float mag35    = sqrtf(vr[idx] * vr[idx] + vi[idx] * vi[idx]);
            float expected = (float)total / 2.0f;
            float rel_err  = fabsf(mag35 - expected) / expected;

            fprintf(stderr,
                    "Verification: bin[3][5] mag=%.1f  expected=%.1f"
                    "  rel_err=%.4f  %s\n",
                    mag35, expected, rel_err,
                    rel_err < 0.01f ? "PASS" : "FAIL");

            fprintf(stderr, "Row-0 bin magnitudes [0..15]:\n");
            for (int k = 0; k < 16; k++) {
                float mag = sqrtf(vr[k] * vr[k] + vi[k] * vi[k]);
                fprintf(stderr, "  [0][%2d] = %.2f\n", k, mag);
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
    double bytes = 2.0 * total * 2 * sizeof(float);
    double GBs   = bytes / (avg_ms / 1000.0) / 1e9;

    printf("cufft_2d,%d,%.3f,%.2f\n", NX, avg_ms, GBs);
    fprintf(stderr,
            "cufft_2d  NX=%d NY=%d  time_ms=%.3f  GB/s=%.2f"
            "  (Accelerate vDSP row+col FFTs)\n",
            NX, NY, avg_ms, GBs);

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                             */
    /* ------------------------------------------------------------------ */
    vDSP_destroy_fftsetup(setup);
    free(realp);
    free(imagp);
    return 0;
}
