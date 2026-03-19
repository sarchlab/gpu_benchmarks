/*
 * int32_throughput.metal
 *
 * Metal compute kernel for INT32 arithmetic throughput benchmark.
 * Used by int32_throughput_metal.mm to measure peak 32-bit integer
 * compute throughput on Apple GPUs.
 *
 * Each thread performs a tight chain of integer multiply-add (MAD) operations
 * using four independent accumulators, keeping the integer pipelines fully
 * busy while hiding per-instruction latency through instruction-level
 * parallelism.
 *
 * The final checksum is written to a one-element output buffer to prevent
 * the compiler from eliminating the computation as dead code.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * int32_mad — tight INT32 MAD loop kernel.
 *
 * @param out              Output buffer (1 int); thread 0 writes the checksum.
 * @param mads_per_thread  Number of MAD iterations per thread (must be a
 *                         multiple of 4 due to the 4× unrolled inner loop).
 * @param tid              Thread index in the dispatch grid.
 */
kernel void int32_mad(device int          *out              [[ buffer(0) ]],
                      constant uint       &mads_per_thread  [[ buffer(1) ]],
                      uint                 tid              [[ thread_position_in_grid ]])
{
    /* Four independent accumulators to exploit instruction-level parallelism
       and hide the INT32 MAD latency on Apple GPU cores. */
    int a0 = 1 + int(tid & 0xFFu);
    int a1 = a0 + 111;
    int a2 = a0 + 222;
    int a3 = a0 + 333;

    /* Multiplier and addend chosen so the compiler cannot constant-fold the
       entire loop (values are non-trivial). */
    const int mul = 3;
    const int add = 1;

    for (uint i = 0; i < mads_per_thread; i += 4) {
        a0 = a0 * mul + add;
        a1 = a1 * mul + add;
        a2 = a2 * mul + add;
        a3 = a3 * mul + add;
    }

    /* Only thread 0 writes to global memory — negligible memory traffic. */
    if (tid == 0) {
        out[0] = a0 + a1 + a2 + a3;
    }
}
