#pragma once
// Complete output reference, preserving the original increasing reduction-index
// order and float arithmetic. Callers provide one element per output.

static void cpu_bicg(const float* A, const float* p, const float* r,
                     float* s_ref, float* q_ref, int m, int n) {
    // s = A^T * r
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++)
            sum += A[i * n + j] * r[i];
        s_ref[j] = sum;
    }
    // q = A * p
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * p[j];
        q_ref[i] = sum;
    }
}
