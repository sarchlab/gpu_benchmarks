/*
 * atomic_operations.metal
 *
 * Metal compute shaders for atomic operation throughput benchmark.
 * Used by atomic_operations_metal.mm to measure atomic throughput
 * on Apple GPUs.
 *
 * Three kernels measure atomicAdd, atomicCAS, and atomicMin throughput
 * on a single global memory location (maximum contention scenario).
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/**
 * atomicAdd_kern — each thread performs ops_per_thread atomic_fetch_add
 * operations on a single device-global atomic counter.
 *
 * @param counter        Shared atomic counter in device memory.
 * @param ops_per_thread Number of atomic operations each thread performs.
 * @param tid            Thread index in the dispatch grid.
 */
kernel void atomicAdd_kern(device atomic_uint  &counter         [[ buffer(0) ]],
                           constant uint       &ops_per_thread  [[ buffer(1) ]],
                           uint                 tid             [[ thread_position_in_grid ]])
{
    for (uint i = 0; i < ops_per_thread; ++i) {
        atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
    }
}

/**
 * atomicCAS_kern — each thread performs ops_per_thread compare-and-swap
 * operations on a single device-global atomic counter.
 *
 * @param counter        Shared atomic counter in device memory.
 * @param ops_per_thread Number of atomic operations each thread performs.
 * @param tid            Thread index in the dispatch grid.
 */
kernel void atomicCAS_kern(device atomic_uint  &counter         [[ buffer(0) ]],
                           constant uint       &ops_per_thread  [[ buffer(1) ]],
                           uint                 tid             [[ thread_position_in_grid ]])
{
    for (uint i = 0; i < ops_per_thread; ++i) {
        uint expected = atomic_load_explicit(&counter, memory_order_relaxed);
        atomic_compare_exchange_weak_explicit(&counter, &expected, expected + 1,
                                             memory_order_relaxed,
                                             memory_order_relaxed);
    }
}

/**
 * atomicMin_kern — each thread performs ops_per_thread atomic_fetch_min
 * operations on a single device-global atomic counter.
 *
 * @param counter        Shared atomic counter in device memory.
 * @param ops_per_thread Number of atomic operations each thread performs.
 * @param tid            Thread index in the dispatch grid.
 */
kernel void atomicMin_kern(device atomic_uint  &counter         [[ buffer(0) ]],
                           constant uint       &ops_per_thread  [[ buffer(1) ]],
                           uint                 tid             [[ thread_position_in_grid ]])
{
    for (uint i = 0; i < ops_per_thread; ++i) {
        atomic_fetch_min_explicit(&counter, tid + i, memory_order_relaxed);
    }
}
