#pragma once
// Frozen numerical GPU bodies from cavazos-lab/PolyBench-ACC
// 70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7
// CUDA/stencils/jacobi-2d-imper/jacobi2D.cu (PolyBench/GPU 1.0).
// N and _PB_N bind to the same runtime n. Bodies differ only in whitespace.
// License: ../../tier2/polybench_jacobi2d/UPSTREAM_LICENSE.
/**
 * jacobi2D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */


namespace pinned_jacobi_gpu {
using DATA_TYPE=float;
#define N n
#define _PB_N n
__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
    {
        B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1 + j)] + A[(1 + i)*N + j] + A[(i-1)*N + j]);
    }
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
    {
        A[i*N + j] = B[i*N + j];
    }
}


#undef N
#undef _PB_N
} // namespace pinned_jacobi_gpu
