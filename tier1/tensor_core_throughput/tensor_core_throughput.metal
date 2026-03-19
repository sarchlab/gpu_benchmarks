/*
 * tensor_core_throughput.metal
 *
 * Metal compute shader for Tensor Core / simdgroup matrix throughput
 * measurement.
 *
 * Uses simdgroup_matrix_multiply_accumulate with 8×8 half-precision matrices
 * to measure matrix multiply throughput on Apple Silicon (M1+).
 *
 * Each simdgroup computes one 8×8 output tile by iterating over K in steps
 * of 8, accumulating via simdgroup_multiply_accumulate.
 *
 * Grid layout:
 *   threadgroups:  (M/8, N/8)
 *   threads per threadgroup:  32 (one simdgroup per threadgroup)
 */

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_gemm(device const half *A          [[ buffer(0) ]],
                           device const half *B          [[ buffer(1) ]],
                           device half       *C          [[ buffer(2) ]],
                           constant uint     &M_val      [[ buffer(3) ]],
                           constant uint     &N_val      [[ buffer(4) ]],
                           constant uint     &K_val      [[ buffer(5) ]],
                           uint2              gid        [[ threadgroup_position_in_grid ]],
                           uint               lane       [[ thread_index_in_simdgroup ]])
{
    const uint TILE = 8;
    uint tileM = gid.x * TILE;
    uint tileN = gid.y * TILE;

    /* Accumulator — 8×8 half-precision result tile. */
    simdgroup_half8x8 acc;
    simdgroup_half8x8 a_frag;
    simdgroup_half8x8 b_frag;

    /* Zero the accumulator. */
    acc = simdgroup_half8x8(0);

    /* Iterate over the K dimension in tiles of 8. */
    for (uint k = 0; k < K_val; k += TILE) {
        /* Load 8×8 sub-matrices from A and B. */
        simdgroup_load(a_frag, A + tileM * K_val + k, K_val);
        simdgroup_load(b_frag, B + k * N_val + tileN, N_val);

        /* Multiply-accumulate. */
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
    }

    /* Store the 8×8 result tile to C. */
    simdgroup_store(acc, C + tileM * N_val + tileN, N_val);
}
