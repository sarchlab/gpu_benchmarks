/*
 * mixbench.metal
 *
 * Metal compute kernel for the parametric roofline (mixbench) benchmark.
 * Used by mixbench_metal.mm to sweep the compute-to-memory ratio.
 *
 * Each thread loads one float (4 bytes) from global memory, performs a
 * variable-length chain of FP32 FMA operations, and stores the result back
 * (4 bytes).  Total memory traffic per thread: 8 bytes.
 *
 * The number of FMA operations (num_fmas) is passed as a buffer constant,
 * allowing the host to sweep different operational intensities without
 * recompiling the shader.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * mixbench_kernel — variable-intensity FMA kernel.
 *
 * @param data          Read/write float array in device memory
 * @param num_elements  Number of float elements in the array
 * @param num_fmas      Number of FMA operations per thread
 * @param tid           Thread index (provided by Metal dispatcher)
 */
kernel void mixbench_kernel(device float       *data          [[ buffer(0) ]],
                            constant uint      &num_elements  [[ buffer(1) ]],
                            constant uint      &num_fmas      [[ buffer(2) ]],
                            uint                tid           [[ thread_position_in_grid ]])
{
    if (tid >= num_elements) {
        return;
    }

    float val = data[tid];

    /* Multiplier and addend chosen so the compiler cannot constant-fold the
       loop.  Values are close to 1 and 0 to avoid overflow/underflow. */
    const float mul = 1.0000001f;
    const float add = 0.0000001f;

    /* Dependency chain: each FMA uses the result of the previous one,
       preventing the compiler from eliminating the computation. */
    for (uint i = 0; i < num_fmas; ++i) {
        val = fma(val, mul, add);
    }

    data[tid] = val;
}
