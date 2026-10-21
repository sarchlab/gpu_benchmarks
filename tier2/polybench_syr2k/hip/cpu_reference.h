#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
// Complete independent double reference for the full, generally nonsymmetric
// result. Transposed input copies make each 4x32 output tile contiguous while
// preserving increasing k order. Inputs remain untouched; scratch is released
// before the caller downloads the measured output.
static std::vector<double> cpu_syr2k(const float* A, const float* B,
                                   const float* initial_C, int n, int m,
                                   float alpha, float beta) {
    std::vector<float> at((size_t)m*n), bt((size_t)m*n);
    for (int i=0; i<n; ++i)
        for (int k=0; k<m; ++k) {
            at[(size_t)k*n+i]=A[(size_t)i*m+k];
            bt[(size_t)k*n+i]=B[(size_t)i*m+k];
        }
    std::vector<double> result((size_t)n*n);
    for (int i=0; i<n; i+=4)
        for (int j=0; j<n; j+=32) {
            const int rows=std::min(4,n-i), cols=std::min(32,n-j);
            double acc[4][32]={};
            for (int k=0; k<m; ++k)
                for (int r=0; r<rows; ++r) {
                    const double a=A[(size_t)(i+r)*m+k];
                    const double b=B[(size_t)(i+r)*m+k];
                    for (int c=0; c<cols; ++c)
                        acc[r][c] += a*bt[(size_t)k*n+j+c] + b*at[(size_t)k*n+j+c];
                }
            for (int r=0; r<rows; ++r)
                for (int c=0; c<cols; ++c) {
                    const size_t q=(size_t)(i+r)*n+j+c;
                    result[q]=(double)alpha*acc[r][c]+(double)beta*initial_C[q];
                }
        }
    return result;
}
// For the declared nonnegative inputs and M <= 2048, 5e-4 exceeds the
// conservative gamma_(3*M+3) FP32 rounding bound (below 3.7e-4). The unit floor
// also admits small zero-reference roundoff. Reject non-finite values first.
static bool syr2k_value_matches(float actual,double expected) {
    const double error=std::fabs((double)actual-expected);
    return std::isfinite(actual) && std::isfinite(expected) && std::isfinite(error)
        && error <= 5e-4*std::max(1.0,std::fabs(expected));
}
