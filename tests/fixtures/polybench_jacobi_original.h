#pragma once
// Frozen numerical bodies from cavazos-lab/PolyBench-ACC
// 70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7
// CUDA/stencils/jacobi-2d-imper/jacobi2D.cu (PolyBench/GPU 1.0).
// Runtime grid views and loop-bound scaffolding are adapted. Numerical bodies
// differ only in whitespace. License: ../../tier2/polybench_jacobi2d/hip/UPSTREAM_LICENSE.
/**
 * jacobi2D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */


#include <vector>
namespace pinned_jacobi {
using DATA_TYPE=float;
struct View {float* data;int n;float* operator[](int i)const{return data+static_cast<size_t>(i)*n;}};
struct Original {
    int N;
    std::vector<float> a,b;
    explicit Original(int n):N(n),a(static_cast<size_t>(n)*n),b(a.size()){init_array(n,{a.data(),n},{b.data(),n});}
#define _PB_N n
#define _PB_TSTEPS tsteps
void init_array(int n, View A, View B)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
            B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
        }
    }
}

void runJacobi2DCpu(int tsteps, int n, View A, View B)
{
    for (int t = 0; t < _PB_TSTEPS; t++)
    {
            for (int i = 1; i < _PB_N - 1; i++)
        {
            for (int j = 1; j < _PB_N - 1; j++)
            {
                B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
            }
        }

            for (int i = 1; i < _PB_N-1; i++)
        {
            for (int j = 1; j < _PB_N-1; j++)
            {
                A[i][j] = B[i][j];
            }
        }
    }
}
#undef _PB_N
#undef _PB_TSTEPS
    void solve(int steps){runJacobi2DCpu(steps,N,{a.data(),N},{b.data(),N});}
};
} // namespace pinned_jacobi
