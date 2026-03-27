/**
 * cufft_2d_metal.mm — 2D complex FFT benchmark using Apple Accelerate vDSP.
 *
 * Performs a 2D forward complex FFT via two passes of 1D FFTs:
 *   1. Apply vDSP_fft_zip to each row  (NY-point FFT, stride=1)
 *   2. Apply vDSP_fft_zip to each column (NX-point FFT, via temp buffer)
 *
 * This is mathematically equivalent to a 2D DFT by the separability property.
 *
 * Parameters (from env vars BENCH_PARAM_*):
 *   grid_dim  — square grid dimension NX=NY for the 2D FFT
 *   warmup    — warmup iterations (not timed)
 *   iters     — timed iterations
 *   direction — FFT direction (forward/inverse/roundtrip)
 *
 * Verification: 2D sinusoidal input at (kx=3, ky=5)
 *   → magnitude at split-complex bin [3][5] ≈ NX*NY/2
 *
 * GB/s = 2 * NX*NY * 2 * sizeof(float) / time_s / 1e9
 *
 * Output (stdout): JSON-lines protocol
 * Output (stderr): human-readable diagnostics
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
 * Compute floor(log2(n)) for positive integers
 * ---------------------------------------------------------------------- */
static vDSP_Length ilog2(int n)
{
    vDSP_Length l = 0;
    while ((1 << (l + 1)) <= n) l++;
    return l;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(void)
{
    @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);

    /* ------------------------------------------------------------------ */
    /* Default parameters (overridden by BENCH_PARAM_* env vars)          */
    /* ------------------------------------------------------------------ */
    int grid_dim = 2048;
    int warmup   = 0;
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

    vDSP_Length log2NX = ilog2(NX);
    vDSP_Length log2NY = ilog2(NY);
    vDSP_Length max_log2 = log2NX > log2NY ? log2NX : log2NY;

    /* Create FFT setup (supports up to length 2^max_log2) */
    FFTSetup setup = vDSP_create_fftsetup(max_log2, FFT_RADIX2);
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
    /* Warmup (not timed)                                                  */
    /* ------------------------------------------------------------------ */
    for (int w = 0; w < warmup; w++) {
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
    for (int iter = 0; iter < iters; iter++) {
        fft_rows(setup, realp, imagp, NX, NY, log2NY);
        fft_cols(setup, realp, imagp, NX, NY, log2NX);
    }
    uint64_t t1 = mach_absolute_time();

    double total_ns = (double)(t1 - t0) * tb.numer / tb.denom;
    double avg_ns   = total_ns / iters;
    double avg_ms   = avg_ns / 1e6;
    double total_ms = total_ns / 1e6;

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

    /* JSON-lines: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"cufftExecC2C\",\"time_ms\":%.6f,"
           "\"params\":{\"grid_dim\":%d,\"warmup\":%d,\"iters\":%d,\"direction\":\"%s\"}}\n",
           avg_ms, grid_dim, warmup, iters, direction);

    /* JSON-lines: summary event */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"GB/s\",\"value\":%.2f}]}\n",
           total_ms, GBs);

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

    } // @autoreleasepool
}
