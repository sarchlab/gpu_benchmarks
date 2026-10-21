#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Complete product with the qualified double arithmetic and increasing k
// order for every output. Reuse contiguous columns across four independent
// rows to bound CPU cost without sampling or changing any reduction.
template<typename Left>
static std::vector<double> cpu_2mm_product(const Left* A, const float* B,
                                          const float* initial, int n,
                                          float alpha, float beta) {
    std::vector<double> result((size_t)n * n);
    for (int i = 0; i < n; i += 4)
        for (int j = 0; j < n; j += 32) {
            const int rows = std::min(4, n-i), cols = std::min(32, n-j);
            double acc[4][32] = {};
            if (initial)
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < cols; ++c)
                        acc[r][c] = beta * (double)initial[(size_t)(i+r)*n+j+c];
            for (int k = 0; k < n; ++k)
                for (int r = 0; r < rows; ++r) {
                    const double left = alpha * (double)A[(size_t)(i+r)*n+k];
                    for (int c = 0; c < cols; ++c)
                        acc[r][c] += left * B[(size_t)k*n+j+c];
                }
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    result[(size_t)(i+r)*n+j+c] = acc[r][c];
        }
    return result;
}

struct CpuTwoMM {
    std::vector<double> tmp;
    std::vector<double> output;
};

// Inputs contain n*n elements for a valid positive extent. The second product
// uses the full independent CPU temporary, never the measured GPU temporary.
static CpuTwoMM cpu_2mm(const float* A, const float* B, const float* C,
                        const float* initial_D, int n, float alpha, float beta) {
    CpuTwoMM ref;
    ref.tmp = cpu_2mm_product(A, B, nullptr, n, alpha, 0.0f);
    ref.output = cpu_2mm_product(ref.tmp.data(), C, initial_D, n, 1.0f, beta);
    return ref;
}

static bool two_mm_value_matches(float actual, double expected) {
    const double error = std::fabs((double)actual - expected);
    return std::isfinite(actual) && std::isfinite(expected) && std::isfinite(error) &&
           error <= 5e-4 * std::max(1.0, std::fabs(expected));
}
