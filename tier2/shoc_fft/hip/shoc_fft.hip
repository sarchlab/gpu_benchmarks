// shoc_fft.hip — SHOC FFT benchmark (HIP, self-contained)
//
// Faithful port of SHOC level1 "fft": BATCHED 512-point complex FFTs, one
// thread block per FFT, 64 threads/block, radix-8 decomposition (three FFT8
// stages) with twiddle factors and shared-memory transposes. The device
// kernels (FFT512_device, IFFT512_device) and all their codelet helpers are
// the ORIGINAL SHOC/Volkov code, taken verbatim from
//   shoc/src/cuda/level1/fft/codelets.h
// and driven with the forward-then-inverse structure of fftlib.cu / FFT.cpp
// runTest. Only the host driver is restructured for the single-run timing
// harness and JSON output.
//
// Timed kernels: FFT512_device (forward) and IFFT512_device (inverse). The
// single timed run is the forward batched FFT over all n_ffts blocks, then
// the inverse batched FFT (which includes the built-in 1/512 normalization,
// verbatim in IFFT512_device). Each is one launch and gets its own record.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./shoc_fft [--n_ffts K] [--precision P]
//     --n_ffts K     Number of independent 512-point FFTs  (default 32768)
//     --precision P  float | double                        (default float)
//
// Environment variables (override argv):
//   BENCH_PARAM_n_ffts, BENCH_PARAM_precision
//
// Output (stdout): one {"type":"kernel",...} JSON line for FFT512_device and
// one for IFFT512_device, each carrying its GPU event time from the single
// timed run, then a summary whose total_time_ms is the SUM of the recorded
// GPU event times. GFLOPS is derived downstream from time_ms and params.
// Output (stderr): human-readable results.

#include "../../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

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

static const char* parseStrParam(int argc, char** argv, const char* name,
                                 const char* defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return defaultVal;
}

// ===========================================================================
// CODELETS — verbatim from SHOC fft/codelets.h (Vasily Volkov, UC Regents).
// ===========================================================================

// Arrange blocks into a 2D grid that fits into the GPU (powers of two only).
inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 )
        slices *= 2;
    return dim3( nblocks/slices, slices );
}

template<class T2, class T> __device__ T2 make_T2(T x, T y);

template <> inline __device__
float2 make_T2<float2,float>(float x, float y) { return make_float2(x, y); }

template <> inline __device__
double2 make_T2<double2,double>(double x, double y) { return make_double2(x, y); }

// complex number arithmetic
template <class T2, class T> inline __device__
T2 cmul( T2 a, T2 b ) { return make_T2<T2,T>( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
template <class T2, class T> inline __device__
T2 cmul( T2 a, T  b ) { return make_T2<T2,T>( b*a.x, b*a.y ); }
template <class T2, class T> inline __device__
T2 cadd( T2 a, T2 b ) { return make_T2<T2,T>( a.x + b.x, a.y + b.y ); }
template <class T2, class T> inline __device__
T2 csub( T2 a, T2 b ) { return make_T2<T2,T>( a.x - b.x, a.y - b.y ); }

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  make_T2<T2,T>(  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  make_T2<T2,T>(  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  make_T2<T2,T>( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  make_T2<T2,T>( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  make_T2<T2,T>( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   make_T2<T2,T>(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   make_T2<T2,T>(  0, -1 )
#define exp_3_8   make_T2<T2,T>( -1, -1 )//requires post-multiply by 1/sqrt(2)

#define iexp_1_16  make_T2<T2,T>(  COS_PI_8,  SIN_PI_8 )
#define iexp_3_16  make_T2<T2,T>(  SIN_PI_8,  COS_PI_8 )
#define iexp_5_16  make_T2<T2,T>( -SIN_PI_8,  COS_PI_8 )
#define iexp_7_16  make_T2<T2,T>( -COS_PI_8,  SIN_PI_8 )
#define iexp_9_16  make_T2<T2,T>( -COS_PI_8, -SIN_PI_8 )
#define iexp_1_8   make_T2<T2,T>(  1, 1 )//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   make_T2<T2,T>(  0, 1 )
#define iexp_3_8   make_T2<T2,T>( -1, 1 )//requires post-multiply by 1/sqrt(2)

template <class T2, class T> inline __device__
T2 exp_i( T phi )
{
    return make_T2<T2,T>( __cosf(phi), __sinf(phi) );
}

// bit reversal
template<int radix> inline __device__ int rev( int bits );
template<> inline __device__ int rev<2>( int bits ) { return bits; }
template<> inline __device__ int rev<4>( int bits )
{
    int reversed[] = {0,2,1,3};
    return reversed[bits];
}
template<> inline __device__ int rev<8>( int bits )
{
    int reversed[] = {0,4,2,6,1,5,3,7};
    return reversed[bits];
}
template<> inline __device__ int rev<16>( int bits )
{
    int reversed[] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    return reversed[bits];
}
inline __device__ int rev4x4( int bits )
{
    int reversed[] = {0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15};
    return reversed[bits];
}

// all FFTs produce output in bit-reversed order
#define IFFT2 FFT2
template<class T2, class T> inline __device__
void FFT2( T2 &a0, T2 &a1 )
{
    T2 c0 = a0;
    a0 = cadd<T2,T>(c0, a1);
    a1 = csub<T2,T>(c0, a1);
}

template<class T2, class T> inline __device__
void FFT4( T2 &a0, T2 &a1, T2 &a2, T2 &a3 )
{
    FFT2<T2,T>( a0, a2 );
    FFT2<T2,T>( a1, a3 );
    a3 = cmul<T2,T>(a3, exp_1_4);
    FFT2<T2,T>( a0, a1 );
    FFT2<T2,T>( a2, a3 );
}

template<class T2, class T> inline __device__
void IFFT4( T2 &a0, T2 &a1, T2 &a2, T2 &a3 )
{
    IFFT2<T2,T>( a0, a2 );
    IFFT2<T2,T>( a1, a3 );
    a3 = cmul<T2,T>(a3, iexp_1_4);
    IFFT2<T2,T>( a0, a1 );
    IFFT2<T2,T>( a2, a3 );
}

template<class T2, class T> inline __device__
void FFT2( T2 *a ) { FFT2<T2,T>( a[0], a[1] ); }
template<class T2, class T> inline __device__
void FFT4( T2 *a ) { FFT4<T2,T>( a[0], a[1], a[2], a[3] ); }
template<class T2, class T> inline __device__
void IFFT4( T2 *a ) { IFFT4<T2,T>( a[0], a[1], a[2], a[3] ); }

template<class T2, class T> inline __device__
void FFT8( T2 *a )
{
    FFT2<T2,T>( a[0], a[4] );
    FFT2<T2,T>( a[1], a[5] );
    FFT2<T2,T>( a[2], a[6] );
    FFT2<T2,T>( a[3], a[7] );

    a[5] = cmul<T2,T>(cmul<T2,T>(a[5], exp_1_8 ), M_SQRT1_2);
    a[6] = cmul<T2,T>(a[6], exp_1_4);
    a[7] = cmul<T2,T>(cmul<T2,T>(a[7], exp_3_8 ), M_SQRT1_2);

    FFT4<T2,T>( a[0], a[1], a[2], a[3] );
    FFT4<T2,T>( a[4], a[5], a[6], a[7] );
}

template<class T2, class T> inline __device__
void IFFT8( T2 *a )
{
    IFFT2<T2,T>( a[0], a[4] );
    IFFT2<T2,T>( a[1], a[5] );
    IFFT2<T2,T>( a[2], a[6] );
    IFFT2<T2,T>( a[3], a[7] );

    a[5] = cmul<T2,T>(cmul<T2,T>(a[5], iexp_1_8 ), M_SQRT1_2);
    a[6] = cmul<T2,T>(a[6], iexp_1_4);
    a[7] = cmul<T2,T>(cmul<T2,T>(a[7], iexp_3_8 ), M_SQRT1_2);

    IFFT4<T2,T>( a[0], a[1], a[2], a[3] );
    IFFT4<T2,T>( a[4], a[5], a[6], a[7] );
}

// loads
template<int n, class T2> inline __device__
void load( T2 *a, T2 *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i] = x[i*sx];
}
template<int n, class T2, class T> inline __device__
void loadx( T2 *a, T *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i].x = x[i*sx];
}
template<int n, class T2, class T> inline __device__
void loady( T2 *a, T *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i].y = x[i*sx];
}

// stores, input is in bit reversed order
template<int n, class T2> inline __device__
void store( T2 *a, T2 *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)];
}
template<int n, class T2, class T> inline __device__
void storex( T2 *a, T *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)].x;
}
template<int n, class T2, class T> inline __device__
void storey( T2 *a, T *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)].y;
}

// multiply by twiddle factors in bit-reversed order
template<int radix, class T2, class T> inline __device__
void twiddle( T2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = cmul<T2,T>(a[j], exp_i<T2,T>((-2*M_PI*rev<radix>(j)/n)*i));
}

template<int radix, class T2, class T> inline __device__
void itwiddle( T2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = cmul<T2,T>(a[j], exp_i<T2,T>((2*M_PI*rev<radix>(j)/n)*i));
}

// transpose via shared memory, input is in bit-reversed layout
template<int n, class T2, class T> inline __device__
void transpose( T2 *a, T *s, int ds, T *l, int dl, int sync = 0xf )
{
    storex<n,T2,T>( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx<n,T2,T> ( a, l, dl );  if( sync&4 ) __syncthreads();
    storey<n,T2,T>( a, s, ds );  if( sync&2 ) __syncthreads();
    loady<n,T2,T> ( a, l, dl );  if( sync&1 ) __syncthreads();
}

template<class T2, class T> __global__
void FFT512_device( T2 *work )
{
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;

    T2 a[8];
    __shared__ T smem[8*8*9];

    load<8, T2>( a, work, 64 );

    FFT8<T2,T>( a );

    twiddle<8,T2,T>( a, tid, 512 );
    transpose<8, T2, T>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );

    FFT8<T2,T>( a );

    twiddle<8,T2,T>( a, hi, 64);
    transpose<8, T2, T>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );

    FFT8<T2,T>( a );

    store<8, T2>( a, work, 64 );
}

template<class T2, class T> __global__
void IFFT512_device( T2 *work )
{
    int i, tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;

    T2 a[8];
    __shared__ T smem[8*8*9];

    load<8, T2>( a, work, 64 );

    IFFT8<T2,T>( a );

    itwiddle<8,T2,T>( a, tid, 512 );
    transpose<8,T2,T>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );

    IFFT8<T2,T>( a );

    itwiddle<8,T2,T>( a, hi, 64);
    transpose<8,T2,T>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );

    IFFT8<T2,T>( a );

    // normalize...
    for (i = 0; i < 8; i++) {
        a[i].x /= 512;
        a[i].y /= 512;
    }
    store<8, T2>( a, work, 64 );
}

// ===========================================================================
// HOST DRIVER — adapted from SHOC fftlib.cu forward()/inverse() + FFT.cpp
// runTest, with each batched launch bracketed for per-kernel timing.
// ===========================================================================

template <class T2, class T>
static int runFFT(int n_ffts, const char* precision)
{
    const int fft_size = 512;
    size_t n_cmplx = (size_t)n_ffts * fft_size;
    size_t bytes   = n_cmplx * sizeof(T2);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "%d batched 512-point FFTs (%s), %.1f MB, block=64\n\n",
            n_ffts, precision, (double)bytes / (1024.0 * 1024.0));

    // Host source (SHOC init: random in [-1,1]) and a working copy.
    T2* source = (T2*)malloc(bytes);
    T2* result = (T2*)malloc(bytes);
    if (!source || !result) { fprintf(stderr, "Host malloc failed\n"); return 1; }
    srand(42);
    for (size_t i = 0; i < n_cmplx; ++i) {
        source[i].x = (T)((rand()/(float)RAND_MAX)*2-1);
        source[i].y = (T)((rand()/(float)RAND_MAX)*2-1);
    }

    T2* work;
    HIP_CHECK(hipMalloc((void**)&work, bytes));

    dim3 grid = grid2D(n_ffts);
    dim3 block(64);

    char params[160];
    snprintf(params, sizeof(params),
             "\"n_ffts\":%d,\"fft_size\":%d,\"precision\":\"%s\"",
             n_ffts, fft_size, precision);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_fwd, acc_inv;

    // The single timed run = forward batched FFT over all n_ffts blocks
    // followed by the inverse batched FFT (IFFT512_device normalizes by 1/512
    // in-kernel, verbatim). The pristine source is uploaded (untimed) before
    // the run so the post-run round-trip check is meaningful.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_fwd);
        bench_lap_reset(&acc_inv);

        HIP_CHECK(hipMemcpy(work, source, bytes, hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc_fwd, bench_time_op(&timer, [&]() {
            FFT512_device<T2,T><<<grid, block>>>(work);
        }));
        bench_lap_add(&acc_inv, bench_time_op(&timer, [&]() {
            IFFT512_device<T2,T><<<grid, block>>>(work);
        }));
        HIP_CHECK(hipGetLastError());
    };

    double total_gpu_ms = 0.0;
    run_pass();
    bench_emit_iter("FFT512_device",  0, &acc_fwd, params);
    bench_emit_iter("IFFT512_device", 0, &acc_inv, params);
    total_gpu_ms += acc_fwd.gpu_ms + acc_inv.gpu_ms;
    fprintf(stderr, "FFT512_device=%.3f ms, IFFT512_device=%.3f ms\n",
            acc_fwd.gpu_ms, acc_inv.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Round-trip verification: forward then inverse (with the built-in /512
    // normalization) must recover the original samples.
    HIP_CHECK(hipMemcpy(result, work, bytes, hipMemcpyDeviceToHost));
    double max_err = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < n_cmplx; ++i) {
        double dx = (double)result[i].x - (double)source[i].x;
        double dy = (double)result[i].y - (double)source[i].y;
        double e = fabs(dx) + fabs(dy);
        if (!std::isfinite(result[i].x) || !std::isfinite(result[i].y) ||
            !std::isfinite(source[i].x) || !std::isfinite(source[i].y) ||
            !std::isfinite(e)) {
            ++nonfinite;
        } else if (e > max_err) {
            max_err = e;
        }
    }
    const bool pass = nonfinite == 0 && max_err < 1e-2;
    fprintf(stderr, "Round-trip max finite abs error: %.3e, non-finite samples: %zu -> %s\n",
            max_err, nonfinite, pass ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(work));
    free(source);
    free(result);
    return pass ? 0 : EXIT_FAILURE;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int n_ffts = 1 << 15; // 32768 independent 512-point FFTs (SHOC-class, scaled)
    const char* precision = "float";

    n_ffts    = parseIntParam(argc, argv, "--n_ffts", n_ffts);
    precision = parseStrParam(argc, argv, "--precision", precision);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_n_ffts");    if (env_val) n_ffts = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision"); if (env_val) precision = env_val;

    if (strcmp(precision, "float") != 0 && strcmp(precision, "double") != 0) {
        fprintf(stderr, "Invalid precision '%s' (use float or double)\n", precision);
        return EXIT_FAILURE;
    }
    if (n_ffts < 1) {
        fprintf(stderr, "n_ffts must be >= 1\n");
        return EXIT_FAILURE;
    }
    if (strcmp(precision, "double") == 0)
        return runFFT<double2, double>(n_ffts, "double");
    return runFFT<float2, float>(n_ffts, "float");
}
