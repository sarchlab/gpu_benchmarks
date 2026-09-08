#pragma once
// Compute only the checked output prefix, preserving each original float
// accumulation order. Callers provide min(8, output dimension) elements.
// This preserves existing coverage; it does not verify the unsampled outputs.

static void cpu_bicg_prefix(const float* A, const float* p, const float* r,
                     float* s_ref, float* q_ref, int m, int n, int check_m, int check_n) {
    // s = A^T * r
    for (int j = 0; j < check_n; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++)
            sum += A[i * n + j] * r[i];
        s_ref[j] = sum;
    }
    // q = A * p
    for (int i = 0; i < check_m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * p[j];
        q_ref[i] = sum;
    }
}
