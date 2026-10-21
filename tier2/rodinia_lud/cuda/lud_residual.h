#ifndef LUD_RESIDUAL_H
#define LUD_RESIDUAL_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lud_reference {
constexpr int row_block = 4;
constexpr int column_block = 64;

// Reconstruct up to row_block rows, starting at first_row, into a row-major
// buffer of row_block*n doubles. The caller supplies 0 <= first_row < n.
// Each output retains the original increasing-k accumulation order.
inline void reconstruct_rows(const float* lu, int n, int first_row, double* output) {
    const int end_row = std::min(n, first_row + row_block);
    for (int first_column = 0; first_column < n; first_column += column_block) {
        const int end_column = std::min(n, first_column + column_block);
        double sums[row_block][column_block] = {};
        for (int k = 0; k <= std::min(end_row - 1, end_column - 1); ++k) {
            for (int i = std::max(first_row, k); i < end_row; ++i) {
                const double l = k == i ? 1.0 : static_cast<double>(lu[static_cast<std::size_t>(i) * n + k]);
                for (int j = std::max(first_column, k); j < end_column; ++j) {
                    const double u = static_cast<double>(lu[static_cast<std::size_t>(k) * n + j]);
                    sums[i - first_row][j - first_column] += l * u;
                }
            }
        }
        for (int i = first_row; i < end_row; ++i) {
            for (int j = first_column; j < end_column; ++j) {
                output[static_cast<std::size_t>(i - first_row) * n + j] = sums[i - first_row][j - first_column];
            }
        }
    }
}

// Full ||A-LU||_F / ||A||_F, using 4*n doubles of scratch. Reducing a complete
// row block only after its reconstruction preserves the original i,j order
// of both norm sums as well as every reconstructed value's k order.
inline double relative_residual(const float* lu, const float* original, int n) {
    std::vector<double> reconstructed(static_cast<std::size_t>(row_block) * n);
    double err = 0.0, norm_a = 0.0;
    for (int first_row = 0; first_row < n; first_row += row_block) {
        reconstruct_rows(lu, n, first_row, reconstructed.data());
        const int end_row = std::min(n, first_row + row_block);
        for (int i = first_row; i < end_row; ++i) {
            for (int j = 0; j < n; ++j) {
                const double sum = reconstructed[static_cast<std::size_t>(i - first_row) * n + j];
                const double a = static_cast<double>(original[static_cast<std::size_t>(i) * n + j]);
                const double diff = a - sum;
                err += diff * diff;
                norm_a += a * a;
            }
        }
    }
    return std::sqrt(err / norm_a);
}
}  // namespace lud_reference

#endif
