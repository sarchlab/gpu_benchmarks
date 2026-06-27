// TODO(bandwidth-rewrite): this file still contains the OLD single-accumulator
// (latency) kernel. Port the new multi-accumulator bandwidth kernel from
// shared_mem_bandwidth.hip. Until then, only the HIP/ROCm backend is correct.
/*
 * shared_mem_bandwidth.metal
 *
 * Metal compute kernels for shared (threadgroup) memory bandwidth
 * measurement.  Two variants:
 *
 *   smem_bw_no_conflict — each thread in a SIMD-group accesses a different
 *                         threadgroup memory bank (stride-1, sequential).
 *   smem_bw_conflict    — all threads in a SIMD-group access the same bank
 *                         (stride-32 between lanes for 32-bank layout).
 *
 * Parameters are passed via a constant struct at buffer(1):
 *   inner_iters  — number of inner-loop iterations
 *   smem_floats  — shared memory array size in floats (e.g. 12288 = 48 KB)
 */

#include <metal_stdlib>
using namespace metal;

// Parameter struct matching the host-side layout
struct Params {
    uint inner_iters;
    uint smem_floats;
    uint threads_per_group;
};

// -------------------------------------------------------------------------
// No-conflict kernel
// -------------------------------------------------------------------------

kernel void smem_bw_no_conflict(
    device float            *d_sink      [[ buffer(0) ]],
    constant Params         &params      [[ buffer(1) ]],
    threadgroup float       *smem        [[ threadgroup(0) ]],
    uint tid                             [[ thread_index_in_threadgroup ]],
    uint tg_id                           [[ threadgroup_position_in_grid ]])
{
    const uint n     = params.threads_per_group;
    const uint count = params.smem_floats;
    const uint iters = params.inner_iters;

    // Initialise threadgroup memory
    for (uint i = tid; i < count; i += n)
        smem[i] = 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint it = 0; it < iters; ++it) {
        // Stride-1 between consecutive threads → different banks
        for (uint i = tid; i < count; i += n) {
            acc += smem[i];
            smem[i] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0)
        d_sink[tg_id] = acc;
}

// -------------------------------------------------------------------------
// Conflict kernel
// -------------------------------------------------------------------------

kernel void smem_bw_conflict(
    device float            *d_sink      [[ buffer(0) ]],
    constant Params         &params      [[ buffer(1) ]],
    threadgroup float       *smem        [[ threadgroup(0) ]],
    uint tid                             [[ thread_index_in_threadgroup ]],
    uint tg_id                           [[ threadgroup_position_in_grid ]])
{
    const uint n     = params.threads_per_group;
    const uint count = params.smem_floats;
    const uint iters = params.inner_iters;
    const uint lane  = tid & 31;
    const uint warp  = tid >> 5;
    const uint nw    = n >> 5;

    // Initialise threadgroup memory (normal stride-1)
    for (uint i = tid; i < count; i += n)
        smem[i] = 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint elems_per_thread = count / n;
    float acc = 0.0f;

    for (uint it = 0; it < iters; ++it) {
        for (uint j = 0; j < elems_per_thread; ++j) {
            // All lanes in a SIMD-group hit the same bank
            uint idx = (lane * 32 + warp + j * nw) % count;
            acc += smem[idx];
            smem[idx] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0)
        d_sink[tg_id] = acc;
}
