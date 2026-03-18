/*
 * cache_latency.metal
 *
 * Metal compute kernel for GPU cache/memory latency measurement.
 * Used by cache_latency_metal.mm.
 *
 * The kernel uses pointer-chasing (dependent index loads) through a
 * pre-built Hamiltonian cycle to expose raw memory access latency.
 * A single thread walks the chain, creating a true data dependency
 * (idx = arr[idx]) that prevents the hardware from overlapping loads.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * pointer_chase — single-thread pointer-chasing latency kernel.
 *
 * Only thread 0 participates; all others return immediately.
 * The serial dependent-load chain directly measures the round-trip
 * latency of the cache/memory level targeted by the array size.
 *
 * @param arr          Index array: arr[i] is the next index to visit.
 * @param params       Packed parameters: [start_idx, num_accesses]
 * @param result       Output: final index (prevents dead-code elimination).
 * @param tid          Thread position in grid (provided by Metal).
 */
kernel void pointer_chase(device const uint  *arr     [[ buffer(0) ]],
                          constant uint2     &params  [[ buffer(1) ]],
                          device       uint  *result  [[ buffer(2) ]],
                          uint tid [[ thread_position_in_grid ]])
{
    // Only thread 0 participates — we measure latency, not throughput.
    if (tid != 0) return;

    uint idx          = params.x;   // start_idx
    uint num_accesses = params.y;   // steps to take

    for (uint i = 0; i < num_accesses; ++i) {
        // Dependent load: the address of the next access depends on
        // the result of the current access.  This serializes all loads.
        idx = arr[idx];
    }

    // Write final index to prevent the compiler from eliminating the loop.
    *result = idx;
}
