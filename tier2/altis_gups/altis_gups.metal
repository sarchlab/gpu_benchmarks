/*
 * altis_gups.metal — Metal compute shader for GUPS benchmark.
 *
 * Each thread performs num_updates random XOR updates on a table of
 * 64-bit entries using xorshift64 PRNG.
 *
 * Buffer layout:
 *   buffer(0): table      — ulong[table_size], the data table
 *   buffer(1): params     — { uint table_size, uint num_updates, ulong seed_base }
 */

#include <metal_stdlib>
using namespace metal;

struct GUPSParams {
    uint   table_size;
    uint   num_updates;
    ulong  seed_base;
};

kernel void gups_kernel(
    device atomic_ulong*    table   [[ buffer(0) ]],
    constant GUPSParams&    params  [[ buffer(1) ]],
    uint                    tid     [[ thread_position_in_grid ]])
{
    // Per-thread xorshift64 state seeded uniquely
    ulong state = params.seed_base + (ulong)tid * 6364136223846793005UL + 1;
    if (state == 0) state = 1;

    uint tsize = params.table_size;

    for (uint u = 0; u < params.num_updates; ++u) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        uint idx = (uint)(state % (ulong)tsize);
        atomic_fetch_xor_explicit(&table[idx], state, memory_order_relaxed);
    }
}
