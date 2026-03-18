/*
 * fp16_throughput.metal
 *
 * Metal compute kernel for FP16 arithmetic throughput measurement.
 * Used by fp16_throughput_metal.mm to measure peak half-precision GFLOPS.
 *
 * Each thread executes a chain of half2 FMA operations using the Metal
 * Shading Language built-in fma() function.  half2 holds two fp16 values,
 * so one fma(half2, half2, half2) call performs two FP16 FMAs simultaneously:
 *
 *     FLOPs per fma(half2, half2, half2) = 2 elements × 2 ops = 4
 *
 * Four independent accumulators per thread hide the FP16 pipeline latency
 * and allow the shader compiler to issue FMA instructions back-to-back.
 *
 * All operands live in registers — only one half2 is written to global memory
 * per kernel invocation (by thread 0 only), so the result reflects pure
 * FP16 arithmetic throughput rather than memory bandwidth.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * fp16_fma — half-precision FMA throughput kernel.
 *
 * @param out              Output buffer (1 half2); receives a checksum from
 *                         thread 0 to prevent dead-code elimination.
 * @param fmas_per_thread  Number of fma() iterations per thread (must be
 *                         a multiple of 4 to match the loop unroll).
 * @param tid              Thread index provided by Metal dispatcher.
 */
kernel void fp16_fma(device half2       *out              [[ buffer(0) ]],
                     constant int       &fmas_per_thread  [[ buffer(1) ]],
                     uint                tid              [[ thread_position_in_grid ]])
{
    /* Initialise four independent half2 accumulators from the thread index
       so the compiler cannot constant-fold the loop. */
    const half base = (half)(1.0f + (float)tid * 0.001f);
    half2 a0 = half2(base,        base + 0.1h);
    half2 a1 = half2(base + 0.2h, base + 0.3h);
    half2 a2 = half2(base + 0.4h, base + 0.5h);
    half2 a3 = half2(base + 0.6h, base + 0.7h);

    /* Multiplier and addend — values near 1 and 0 prevent saturation while
       staying far enough from zero that the compiler cannot fold the loop. */
    const half2 mul = half2(1.0000001h, 1.0000001h);
    const half2 add = half2(0.0000001h, 0.0000001h);

    /* Main compute loop, unrolled by 4.
       Each fma(half2, half2, half2) delivers 4 FLOPs. */
    for (int i = 0; i < fmas_per_thread; i += 4) {
        a0 = fma(a0, mul, add);
        a1 = fma(a1, mul, add);
        a2 = fma(a2, mul, add);
        a3 = fma(a3, mul, add);
    }

    /* Write a reduction of all accumulators to global memory so the compiler
       cannot elide the computation.  Only thread 0 writes, so memory traffic
       is negligible. */
    if (tid == 0u) {
        out[0] = (a0 + a1) + (a2 + a3);
    }
}
