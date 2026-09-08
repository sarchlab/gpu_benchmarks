#pragma once
// Compute only the checked output prefix, preserving each original float
// accumulation order. Callers provide min(8, output dimension) elements.
// This preserves existing coverage; it does not verify the unsampled outputs.

static void cpu_mvt_prefix(const float* A, const float* y1, const float* y2,
                    float* x1_ref, float* x2_ref, int n, int check_n) {
    // x1 = A * y1
    for (int i = 0; i < check_n; i++) {
        x1_ref[i] = 0.0f;
        for (int j = 0; j < n; j++)
            x1_ref[i] += A[i * n + j] * y1[j];
    }
    // x2 = A^T * y2
    for (int j = 0; j < check_n; j++) {
        x2_ref[j] = 0.0f;
        for (int i = 0; i < n; i++)
            x2_ref[j] += A[i * n + j] * y2[i];
    }
}
