#pragma once
// Complete output reference, preserving the original increasing reduction-index
// order and float arithmetic. Callers provide one element per output.

static void cpu_mvt(const float* A, const float* y1, const float* y2,
                    float* x1_ref, float* x2_ref, int n) {
    // x1 = A * y1
    for (int i = 0; i < n; i++) {
        x1_ref[i] = 0.0f;
        for (int j = 0; j < n; j++)
            x1_ref[i] += A[i * n + j] * y1[j];
    }
    // x2 = A^T * y2
    for (int j = 0; j < n; j++) {
        x2_ref[j] = 0.0f;
        for (int i = 0; i < n; i++)
            x2_ref[j] += A[i * n + j] * y2[i];
    }
}
