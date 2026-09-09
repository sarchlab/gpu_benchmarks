// npb_ep.hip — NAS Parallel Benchmarks Embarrassingly Parallel (EP)
//
// Faithful port of NPB-GPU (GMAP) EP: the device kernel (gpu_kernel) and its
// random-number generators (randlc_device / vranlc_device) are the ORIGINAL
// upstream source. EP generates 2^(M+1) pseudorandom numbers via the NAS 46-bit
// linear congruential generator (randlc, R23/R46 split), forms Gaussian deviate
// pairs by the Marsaglia polar (acceptance-rejection) method, tallies them into
// 10 concentric square annuli, and accumulates the Sx/Sy sums. Purely
// compute-bound. Only the host driver is restructured for per-iteration timing.
//
// Upstream kernel/RNG taken verbatim (modulo the two notes below) from:
//   NPB-GPU/hip/EP/ep.cpp   (gpu_kernel, randlc_device, vranlc_device)
//   NPB-GPU/hip/common/npb.hpp (host randlc, R23/R46/T23/T46)
// Two minimal faithful deviations:
//   1. The compile-time NN macro (grid guard) is passed to the kernel as `nn`
//      so the class M can be a runtime parameter instead of requiring a
//      recompile (the original fixes M in npbparams.hpp).
//   2. Nothing else — math, rejection test, binning and reduction are verbatim.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
// Uses double precision (NAS spec). Double atomicAdd needs sm_60+ / any GCN.
//
// Class is set by M: M=24 -> S, 25 -> W, 28 -> A, 30 -> B, 32 -> C.
// Default M=28 (Class A): NN = 2^12 threads, each doing NK = 2^16 numbers,
// i.e. 2^28 pairs / 2^29 numbers per launch — runs in well under a second on a
// modern GPU yet gives a substantial (>~ms) per-pass kernel time.
//
// Usage:
//   ./npb_ep [--m M]
//
// Environment variables (override argv):
//   BENCH_PARAM_m           (class exponent M, default 28 = Class A)
//   BENCH_PARAM_block_size  (threads per block, default 256)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed pass carrying
// the GPU event time and host wall-clock time of the single gpu_kernel launch,
// then a summary whose total_time_ms is the SUM of all passes' GPU event times.
// Output (stderr): human-readable EP results + NAS verification.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// NAS EP constants — verbatim from NPB-GPU EP/ep.cpp and common/npb.hpp
// ---------------------------------------------------------------------------
#define MK            (16)
#define NK            (1 << MK)            // numbers per batch group = 2^16 (fixed)
#define NQ            (10)                 // number of annular bins
#define EPSILON       (1.0e-8)
#define A             (1220703125.0)
#define S             (271828183.0)
#define RECOMPUTATION (128)

// R23 = 2^-23, T23 = 2^23, R46 = 2^-46, T46 = 2^46 (literal products; the
// upstream defines these via pow — we use the equivalent product form so the
// values are compile-time constants usable in both host and device code).
#define R23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define R46 (R23*R23)
#define T23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define T46 (T23*T23)

// ---------------------------------------------------------------------------
// RNG — verbatim from NPB-GPU EP/ep.cpp (__device__ versions)
// ---------------------------------------------------------------------------
__device__ double randlc_device(double* x,
        double a){
    double t1,t2,t3,t4,a1,a2,x1,x2,z;
    t1 = R23 * a;
    a1 = (int)t1;
    a2 = a - T23 * a1;
    t1 = R23 * (*x);
    x1 = (int)t1;
    x2 = (*x) - T23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int)(R23 * t1);
    z = t1 - T23 * t2;
    t3 = T23 * z + a2 * x2;
    t4 = (int)(R46 * t3);
    (*x) = t3 - T46 * t4;
    return (R46 * (*x));
}

__device__ void vranlc_device(int n,
        double* x_seed,
        double a,
        double* y){
    int i;
    double x,t1,t2,t3,t4,a1,a2,x1,x2,z;
    t1 = R23 * a;
    a1 = (int)t1;
    a2 = a - T23 * a1;
    x = *x_seed;
    for(i=0; i<n; i++){
        t1 = R23 * x;
        x1 = (int)t1;
        x2 = x - T23 * x1;
        t1 = a1 * x2 + a2 * x1;
        t2 = (int)(R23 * t1);
        z = t1 - T23 * t2;
        t3 = T23 * z + a2 * x2;
        t4 = (int)(R46 * t3);
        x = t3 - T46 * t4;
        y[i] = R46 * x;
    }
    *x_seed = x;
}

// Host randlc — verbatim from NPB-GPU common/npb.hpp (used to derive `an`)
static double randlc(double* x, double a){
    double t1,t2,t3,t4,a1,a2,x1,x2,z;
    t1 = R23 * a;
    a1 = (int)t1;
    a2 = a - T23 * a1;
    t1 = R23 * (*x);
    x1 = (int)t1;
    x2 = (*x) - T23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int)(R23 * t1);
    z = t1 - T23 * t2;
    t3 = T23 * z + a2 * x2;
    t4 = (int)(R46 * t3);
    (*x) = t3 - T46 * t4;
    return (R46 * (*x));
}

// ---------------------------------------------------------------------------
// EP kernel — verbatim from NPB-GPU EP/ep.cpp gpu_kernel.
// Only faithful change: the compile-time NN grid guard is a runtime arg `nn`
// so class M is selectable without recompiling.
// ---------------------------------------------------------------------------
__global__ void gpu_kernel(double* q_global,
        double* sx_global,
        double* sy_global,
        double an,
        int nn){
    double x_local[2*RECOMPUTATION];
    double q_local[NQ];
    double sx_local, sy_local;
    double t1, t2, t3, t4, x1, x2, seed;
    int i, ii, ik, kk, l;

    q_local[0]=0.0;
    q_local[1]=0.0;
    q_local[2]=0.0;
    q_local[3]=0.0;
    q_local[4]=0.0;
    q_local[5]=0.0;
    q_local[6]=0.0;
    q_local[7]=0.0;
    q_local[8]=0.0;
    q_local[9]=0.0;
    sx_local=0.0;
    sy_local=0.0;

    kk=blockIdx.x*blockDim.x+threadIdx.x;

    if(kk>=nn){return;}

    t1=S;
    t2=an;

    /* find starting seed t1 for this kk */
    for(i=1; i<=100; i++){
        ik=kk/2;
        if((2*ik)!=kk){t3=randlc_device(&t1, t2);}
        if(ik==0){break;}
        t3=randlc_device(&t2, t2);
        kk=ik;
    }

    seed=t1;
    for(ii=0; ii<NK; ii=ii+RECOMPUTATION){
        /* compute uniform pseudorandom numbers */
        vranlc_device(2*RECOMPUTATION, &seed, A, x_local);

        /*
         * compute gaussian deviates by acceptance-rejection method and
         * tally counts in concentric square annuli. this loop is not
         * vectorizable.
         */
        for(i=0; i<RECOMPUTATION; i++){
            x1=2.0*x_local[2*i]-1.0;
            x2=2.0*x_local[2*i+1]-1.0;
            t1=x1*x1+x2*x2;
            if(t1<=1.0){
                t2=sqrt(-2.0*log(t1)/t1);
                t3=(x1*t2);
                t4=(x2*t2);
                l=max(fabs(t3), fabs(t4));
                q_local[l]+=1.0;
                sx_local+=t3;
                sy_local+=t4;
            }
        }
    }

    atomicAdd(q_global+blockIdx.x*NQ+0, q_local[0]);
    atomicAdd(q_global+blockIdx.x*NQ+1, q_local[1]);
    atomicAdd(q_global+blockIdx.x*NQ+2, q_local[2]);
    atomicAdd(q_global+blockIdx.x*NQ+3, q_local[3]);
    atomicAdd(q_global+blockIdx.x*NQ+4, q_local[4]);
    atomicAdd(q_global+blockIdx.x*NQ+5, q_local[5]);
    atomicAdd(q_global+blockIdx.x*NQ+6, q_local[6]);
    atomicAdd(q_global+blockIdx.x*NQ+7, q_local[7]);
    atomicAdd(q_global+blockIdx.x*NQ+8, q_local[8]);
    atomicAdd(q_global+blockIdx.x*NQ+9, q_local[9]);
    atomicAdd(sx_global+blockIdx.x, sx_local);
    atomicAdd(sy_global+blockIdx.x, sy_local);
}

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

static const char* class_of(int m) {
    switch (m) {
        case 24: return "S";
        case 25: return "W";
        case 28: return "A";
        case 30: return "B";
        case 32: return "C";
        case 36: return "D";
        case 40: return "E";
        default: return "?";
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Default M=28 => Class A.
    int m          = parseIntParam(argc, argv, "--m", 28);
    int block_size = 256;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_m");          if (env_val) m = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size"); if (env_val) block_size = atoi(env_val);

    if (m <= MK) {
        fprintf(stderr, "M must be > %d (M-MK = number-of-groups exponent must be >= 1)\n", MK);
        return EXIT_FAILURE;
    }
    if (m - MK >= std::numeric_limits<int>::digits) {
        fprintf(stderr, "M produces a thread count that cannot fit int\n");
        return EXIT_FAILURE;
    }
    if (block_size <= 0) {
        fprintf(stderr, "block_size must be positive\n");
        return EXIT_FAILURE;
    }

    // Grid geometry (original setup_gpu): NN = 2^(M-MK) total threads.
    long long nn = 1LL << (m - MK);
    int threads_per_block = block_size;
    int blocks_per_grid   = (int)((nn + threads_per_block - 1) / threads_per_block);
    if ((long long)blocks_per_grid * NQ - 1 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Per-block histogram index cannot fit int\n");
        return EXIT_FAILURE;
    }

    // an = A^(2^MK): MK+1 randlc iterations from A (original main setup).
    double t1 = A;
    for (int i = 0; i < MK + 1; ++i) {
        randlc(&t1, t1);
    }
    double an = t1;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "NAS EP Class %s  |  M=%d  |  N = 2^%d numbers  |  NN = %lld threads\n",
            class_of(m), m, m + 1, nn);
    fprintf(stderr, "Grid: %d blocks x %d threads\n\n",
            blocks_per_grid, threads_per_block);

    // Device accumulators (per-block partial sums).
    size_t size_q  = (size_t)blocks_per_grid * NQ * sizeof(double);
    size_t size_sx = (size_t)blocks_per_grid * sizeof(double);
    size_t size_sy = (size_t)blocks_per_grid * sizeof(double);
    double* q_device;
    double* sx_device;
    double* sy_device;
    HIP_CHECK(hipMalloc(&q_device,  size_q));
    HIP_CHECK(hipMalloc(&sx_device, size_sx));
    HIP_CHECK(hipMalloc(&sy_device, size_sy));

    char params[128];
    snprintf(params, sizeof(params),
             "\"class\":\"%s\",\"m\":%d,\"block_size\":%d",
             class_of(m), m, block_size);

    // One recorded pass = one gpu_kernel launch (EP is embarrassingly parallel;
    // the per-block partial reduction is inside the kernel). The RNG seeds are
    // derived from fixed constants inside the kernel, so every launch performs
    // identical work — no input reset needed. The output accumulators, however,
    // are written via atomicAdd, so they are zeroed (untimed, OUTSIDE the timed
    // bracket) before each launch to keep the final NAS verification meaningful.
    bench_timer_t timer;
    bench_timer_create(&timer);

    auto run_launch = [&](bench_lap_t* lap_out) {
        HIP_CHECK(hipMemset(q_device,  0, size_q));
        HIP_CHECK(hipMemset(sx_device, 0, size_sx));
        HIP_CHECK(hipMemset(sy_device, 0, size_sy));
        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();
        *lap_out = bench_time_op(&timer, [&]() {
            gpu_kernel<<<dim3(blocks_per_grid), dim3(threads_per_block)>>>(
                q_device, sx_device, sy_device, an, (int)nn);
        });
        HIP_CHECK(hipGetLastError());
    };

    bench_lap_t lap;
    run_launch(&lap);
    double total_gpu_ms = 0.0;
    bench_emit_iter("gpu_kernel", 0, &lap, params);
    total_gpu_ms += lap.gpu_ms;
    fprintf(stderr, "gpu_kernel: gpu = %.3f ms, host = %.3f ms\n",
            lap.gpu_ms, lap.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // Host-side final reduction + NAS verification (untimed).
    // The device buffers hold the LAST launch's per-block partials.
    // -------------------------------------------------------------------
    double* q_host  = (double*)malloc(size_q);
    double* sx_host = (double*)malloc(size_sx);
    double* sy_host = (double*)malloc(size_sy);
    HIP_CHECK(hipMemcpy(q_host,  q_device,  size_q,  hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(sx_host, sx_device, size_sx, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(sy_host, sy_device, size_sy, hipMemcpyDeviceToHost));

    double q[NQ];
    for (int i = 0; i < NQ; ++i) q[i] = 0.0;
    double sx = 0.0, sy = 0.0;
    for (int block = 0; block < blocks_per_grid; ++block) {
        for (int i = 0; i < NQ; ++i) q[i] += q_host[block*NQ + i];
        sx += sx_host[block];
        sy += sy_host[block];
    }
    double gc = 0.0;
    for (int i = 0; i < NQ; ++i) gc += q[i];

    // NAS reference sums per class (from NPB-GPU EP/ep.cpp).
    double sx_verify = 0.0, sy_verify = 0.0;
    bool have_ref = true;
    if      (m == 24) { sx_verify = -3.247834652034740e+3; sy_verify = -6.958407078382297e+3; }
    else if (m == 25) { sx_verify = -2.863319731645753e+3; sy_verify = -6.320053679109499e+3; }
    else if (m == 28) { sx_verify = -4.295875165629892e+3; sy_verify = -1.580732573678431e+4; }
    else if (m == 30) { sx_verify =  4.033815542441498e+4; sy_verify = -2.660669192809235e+4; }
    else if (m == 32) { sx_verify =  4.764367927995374e+4; sy_verify = -8.084072988043731e+4; }
    else if (m == 36) { sx_verify =  1.982481200946593e+5; sy_verify = -1.020596636361769e+5; }
    else if (m == 40) { sx_verify = -5.319717441530e+05;   sy_verify = -3.688834557731e+05;   }
    else              { have_ref = false; }

    bool verified = have_ref;
    if (have_ref) {
        double sx_err = fabs((sx - sx_verify) / sx_verify);
        double sy_err = fabs((sy - sy_verify) / sy_verify);
        verified = (sx_err <= EPSILON) && (sy_err <= EPSILON);
    }

    fprintf(stderr, "\nNAS EP Results (Class %s, M=%d):\n", class_of(m), m);
    fprintf(stderr, "  N = 2^%d\n", m + 1);
    fprintf(stderr, "  No. Gaussian Pairs = %15.0f\n", gc);
    fprintf(stderr, "  Sums = %25.15e %25.15e\n", sx, sy);
    fprintf(stderr, "  Counts:\n");
    for (int i = 0; i < NQ; ++i) fprintf(stderr, "    %3d %15.0f\n", i, q[i]);
    if (!have_ref)      fprintf(stderr, "  No reference sums for M=%d (verification skipped)\n", m);
    else if (verified)  fprintf(stderr, "  Verification: PASS\n");
    else                fprintf(stderr, "  Verification: FAIL (sx=%.15e vs %.15e, sy=%.15e vs %.15e)\n",
                                 sx, sx_verify, sy, sy_verify);

    free(q_host);
    free(sx_host);
    free(sy_host);
    HIP_CHECK(hipFree(q_device));
    HIP_CHECK(hipFree(sx_device));
    HIP_CHECK(hipFree(sy_device));

    // Unknown M has no reference sums (verification skipped): exit 0.
    // A known class whose sums mismatch is a wrong computation: exit non-zero.
    return (!have_ref || verified) ? EXIT_SUCCESS : EXIT_FAILURE;
}
