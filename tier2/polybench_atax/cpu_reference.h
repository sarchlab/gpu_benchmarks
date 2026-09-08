#pragma once
// Compute only the checked output prefix, preserving each original float
// accumulation order. Callers provide min(8, output dimension) elements.
// This preserves existing coverage; it does not verify the unsampled outputs.
#include <vector>

static void cpu_atax_prefix(const float* A, const float* x, float* y_ref, int nx, int ny, int check_n) {
    // tmp = A * x
    std::vector<float> tmp(nx, 0.0f);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            tmp[i] += A[i * ny + j] * x[j];
    // y = A^T * tmp
    for (int j = 0; j < check_n; j++) {
        y_ref[j] = 0.0f;
        for (int i = 0; i < nx; i++)
            y_ref[j] += A[i * ny + j] * tmp[i];
    }
}
