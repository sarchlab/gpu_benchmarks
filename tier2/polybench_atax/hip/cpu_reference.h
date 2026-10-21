#pragma once
// Complete output reference, preserving the original increasing reduction-index
// order and float arithmetic. Callers provide one element per output.
#include <vector>

static void cpu_atax(const float* A, const float* x, float* y_ref, int nx, int ny) {
    // tmp = A * x
    std::vector<float> tmp(nx, 0.0f);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            tmp[i] += A[i * ny + j] * x[j];
    // y = A^T * tmp
    for (int j = 0; j < ny; j++) {
        y_ref[j] = 0.0f;
        for (int i = 0; i < nx; i++)
            y_ref[j] += A[i * ny + j] * tmp[i];
    }
}
