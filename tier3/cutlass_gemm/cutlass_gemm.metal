/**
 * cutlass_gemm.metal — CUTLASS-style tiled FP32 GEMM Metal compute shader.
 *
 * Computes: C = A * B  (row-major, FP32)
 *   where A is M×K, B is K×N, C is M×N
 *
 * Tile parameters:
 *   Block tile:  BM×BN = 64×64
 *   K-tile:      BK = 8
 *   Thread tile: TM×TN = 8×8
 *   Threads per threadgroup: 64 (8×8)
 *
 * Algorithm:
 *   Each threadgroup computes a BM×BN tile of C.
 *   For each K-tile:
 *     1. Cooperatively load BM×BK of A and BK×BN of B into threadgroup memory
 *     2. Each thread computes a TM×TN sub-tile using register tiling
 *        (outer-product accumulation over the BK dimension)
 *     3. Barrier before the next K-tile
 *   Finally, each thread writes its TM×TN results to device memory.
 */

#include <metal_stdlib>
using namespace metal;

/* Kernel parameters passed from host */
struct GemmParams {
    uint M;
    uint N;
    uint K;
};

/* Tile sizes — must match host-side constants */
constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 8;
constant uint TM = 8;
constant uint TN = 8;
constant uint NUM_THREADS = (BM / TM) * (BN / TN);  /* 64 */

kernel void tiled_gemm(
    device const float*    A       [[buffer(0)]],
    device const float*    B       [[buffer(1)]],
    device float*          C       [[buffer(2)]],
    constant GemmParams&   params  [[buffer(3)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    /* Threadgroup (shared) memory for A and B tiles */
    threadgroup float smA[64][8];   /* BM × BK */
    threadgroup float smB[8][64];   /* BK × BN */

    /* Block position in the output grid */
    uint bx = tg_pos.x;   /* column block index */
    uint by = tg_pos.y;   /* row block index */

    /* Thread position within the 8×8 thread tile grid */
    uint tx = tid % (BN / TN);   /* 0..7 */
    uint ty = tid / (BN / TN);   /* 0..7 */

    /* Accumulator registers — each thread computes TM×TN outputs */
    float regC[8][8];   /* TM × TN */
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            regC[i][j] = 0.0f;

    /* Loop over K dimension in tiles of BK */
    for (uint bk = 0; bk < K; bk += BK) {

        /* ---- Load A tile [BM × BK] into threadgroup memory ---- */
        /* 512 elements / 64 threads = 8 elements per thread */
        for (uint load = 0; load < (BM * BK) / NUM_THREADS; load++) {
            uint idx   = load * NUM_THREADS + tid;
            uint smRow = idx / BK;
            uint smCol = idx % BK;
            uint gRow  = by * BM + smRow;
            uint gCol  = bk + smCol;
            smA[smRow][smCol] = (gRow < M && gCol < K)
                                ? A[gRow * K + gCol]
                                : 0.0f;
        }

        /* ---- Load B tile [BK × BN] into threadgroup memory ---- */
        /* 512 elements / 64 threads = 8 elements per thread */
        for (uint load = 0; load < (BK * BN) / NUM_THREADS; load++) {
            uint idx   = load * NUM_THREADS + tid;
            uint smRow = idx / BN;
            uint smCol = idx % BN;
            uint gRow  = bk + smRow;
            uint gCol  = bx * BN + smCol;
            smB[smRow][smCol] = (gRow < K && gCol < N)
                                ? B[gRow * N + gCol]
                                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* ---- Compute thread tile via outer-product accumulation ---- */
        for (uint kk = 0; kk < BK; kk++) {
            /* Load A column and B row into registers */
            float regA[8], regB[8];   /* TM, TN */
            for (uint i = 0; i < TM; i++)
                regA[i] = smA[ty * TM + i][kk];
            for (uint j = 0; j < TN; j++)
                regB[j] = smB[kk][tx * TN + j];

            /* Outer product */
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    regC[i][j] += regA[i] * regB[j];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* ---- Write results to device memory ---- */
    for (uint i = 0; i < TM; i++) {
        uint gRow = by * BM + ty * TM + i;
        if (gRow < M) {
            for (uint j = 0; j < TN; j++) {
                uint gCol = bx * BN + tx * TN + j;
                if (gCol < N)
                    C[gRow * N + gCol] = regC[i][j];
            }
        }
    }
}
