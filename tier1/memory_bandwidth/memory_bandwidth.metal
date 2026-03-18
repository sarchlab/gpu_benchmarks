/*
 * memory_bandwidth.metal
 *
 * Metal compute kernel for device-to-device (D2D) memory copy.
 * Used by memory_bandwidth_metal.mm to measure GPU internal memory bandwidth.
 *
 * The kernel copies one float4 (16 bytes) per thread, yielding a wide
 * vectorized load/store pattern that saturates memory bandwidth.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * d2d_copy — vectorized device-to-device memory copy kernel.
 *
 * @param src       Source buffer (read-only device memory)
 * @param dst       Destination buffer (read-write device memory)
 * @param num_f4    Number of float4 elements to copy
 * @param tid       Thread index (provided by Metal dispatcher)
 */
kernel void d2d_copy(device const float4 *src [[ buffer(0) ]],
                     device       float4 *dst [[ buffer(1) ]],
                     constant uint       &num_f4 [[ buffer(2) ]],
                     uint tid [[ thread_position_in_grid ]])
{
    if (tid < num_f4) {
        dst[tid] = src[tid];
    }
}
