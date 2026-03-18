/*
 * fp32_throughput.metal
 *
 * Metal compute kernel for FP32 arithmetic throughput benchmark.
 * Used by fp32_throughput_metal.mm to measure peak single-precision
 * floating-point compute throughput on Apple GPUs.
 *
 * Each thread performs a tight chain of fused multiply-add (FMA) operations
 * using four independent accumulators, keeping the FP pipelines fully busy
 * while hiding per-instruction latency through instruction-level parallelism.
 *
 * The final checksum is written to a one-element output buffer to prevent
 * the compiler from eliminating the computation as dead code.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * fp32_fma — tight FP32 FMA loop kernel.
 *
 * @param out             Output buffer (1 float); thread 0 writes the checksum.
 * @param fmas_per_thread Number of FMA iterations per thread (must be a
 *                        multiple of 4 due to the 4× unrolled inner loop).
 * @param tid             Thread index in the dispatch grid.
 */
kernel void fp32_fma(device float       *out              [[ buffer(0) ]],
                     constant uint      &fmas_per_thread  [[ buffer(1) ]],
                     uint                tid              [[ thread_position_in_grid ]])
{
    /* Four independent accumulators to exploit instruction-level parallelism
       and hide the FP32 FMA latency on Apple GPU cores. */
    float a0 = 1.0f + float(tid & 0xFFu) * 0.001f;
    float a1 = a0 + 0.1f;
    float a2 = a0 + 0.2f;
    float a3 = a0 + 0.3f;

    /* Multiplier and addend chosen so the compiler cannot constant-fold the
       entire loop (values are very close to 1 and 0 respectively). */
    const float mul = 1.0000001f;
    const float add = 0.0000001f;

    for (uint i = 0; i < fmas_per_thread; i += 4) {
        a0 = fma(a0, mul, add);
        a1 = fma(a1, mul, add);
        a2 = fma(a2, mul, add);
        a3 = fma(a3, mul, add);
    }

    /* Only thread 0 writes to global memory — negligible memory traffic. */
    if (tid == 0) {
        out[0] = a0 + a1 + a2 + a3;
    }
}
