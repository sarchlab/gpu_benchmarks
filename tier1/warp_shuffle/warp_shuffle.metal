/*
 * warp_shuffle.metal
 *
 * Metal compute shaders for warp/SIMD shuffle throughput benchmark.
 * Used by warp_shuffle_metal.mm to measure shuffle throughput on Apple GPUs.
 *
 * Four kernels measure the throughput of the SIMD shuffle intrinsics:
 *   - simd_shuffle_xor   (butterfly / XOR shuffle)
 *   - simd_shuffle        (indexed shuffle)
 *   - simd_shuffle_up     (shift up)
 *   - simd_shuffle_down   (shift down)
 *
 * Each thread performs ops_per_thread shuffle operations in a tight loop,
 * creating a dependency chain to prevent dead-code elimination.
 * Only thread 0 writes the final result to prevent memory traffic from
 * dominating the measurement.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

/**
 * shfl_xor_kern — butterfly/XOR SIMD shuffle.
 *
 * @param out             Output buffer (1 int); thread 0 writes the result.
 * @param ops_per_thread  Number of shuffle iterations per thread.
 * @param tid             Thread index in the dispatch grid.
 * @param lane            Lane index within the SIMD group (0–31).
 */
kernel void shfl_xor_kern(device int        *out              [[ buffer(0) ]],
                          constant uint     &ops_per_thread   [[ buffer(1) ]],
                          uint               tid              [[ thread_position_in_grid ]],
                          uint               lane             [[ thread_index_in_simdgroup ]])
{
    int val = (int)tid;
    for (uint i = 0; i < ops_per_thread; ++i) {
        val = simd_shuffle_xor(val, 1);
    }
    if (tid == 0) {
        out[0] = val;
    }
}

/**
 * shfl_kern — indexed (direct) SIMD shuffle.
 *
 * @param out             Output buffer (1 int); thread 0 writes the result.
 * @param ops_per_thread  Number of shuffle iterations per thread.
 * @param tid             Thread index in the dispatch grid.
 * @param lane            Lane index within the SIMD group (0–31).
 */
kernel void shfl_kern(device int        *out              [[ buffer(0) ]],
                      constant uint     &ops_per_thread   [[ buffer(1) ]],
                      uint               tid              [[ thread_position_in_grid ]],
                      uint               lane             [[ thread_index_in_simdgroup ]])
{
    int val = (int)tid;
    ushort src_lane = (ushort)((lane ^ 1) % 32);
    for (uint i = 0; i < ops_per_thread; ++i) {
        val = simd_shuffle(val, src_lane);
    }
    if (tid == 0) {
        out[0] = val;
    }
}

/**
 * shfl_up_kern — shift-up SIMD shuffle.
 *
 * @param out             Output buffer (1 int); thread 0 writes the result.
 * @param ops_per_thread  Number of shuffle iterations per thread.
 * @param tid             Thread index in the dispatch grid.
 * @param lane            Lane index within the SIMD group (0–31).
 */
kernel void shfl_up_kern(device int        *out              [[ buffer(0) ]],
                         constant uint     &ops_per_thread   [[ buffer(1) ]],
                         uint               tid              [[ thread_position_in_grid ]],
                         uint               lane             [[ thread_index_in_simdgroup ]])
{
    int val = (int)tid;
    for (uint i = 0; i < ops_per_thread; ++i) {
        val = simd_shuffle_up(val, 1);
    }
    if (tid == 0) {
        out[0] = val;
    }
}

/**
 * shfl_down_kern — shift-down SIMD shuffle.
 *
 * @param out             Output buffer (1 int); thread 0 writes the result.
 * @param ops_per_thread  Number of shuffle iterations per thread.
 * @param tid             Thread index in the dispatch grid.
 * @param lane            Lane index within the SIMD group (0–31).
 */
kernel void shfl_down_kern(device int        *out              [[ buffer(0) ]],
                           constant uint     &ops_per_thread   [[ buffer(1) ]],
                           uint               tid              [[ thread_position_in_grid ]],
                           uint               lane             [[ thread_index_in_simdgroup ]])
{
    int val = (int)tid;
    for (uint i = 0; i < ops_per_thread; ++i) {
        val = simd_shuffle_down(val, 1);
    }
    if (tid == 0) {
        out[0] = val;
    }
}
