#pragma once

#include <cstddef>
#include <vector>

// Existing qualified float operator. Only the interior is updated; callers
// initialize the output boundary before the first step.
static void cpu_stencil2d(const float* in, float* out, int N)
{
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            float center = in[i * N + j];
            float north  = in[(i - 1) * N + j];
            float south  = in[(i + 1) * N + j];
            float west   = in[i * N + (j - 1)];
            float east   = in[i * N + (j + 1)];
            float nw     = in[(i - 1) * N + (j - 1)];
            float ne     = in[(i - 1) * N + (j + 1)];
            float sw     = in[(i + 1) * N + (j - 1)];
            float se     = in[(i + 1) * N + (j + 1)];

            out[i * N + j] = 0.5f   * center
                           + 0.1f   * (north + south + west + east)
                           + 0.025f * (nw + ne + sw + se);
        }
    }
}

// Complete reference from the original input, retaining the existing per-cell
// arithmetic and fixed boundaries through every requested timestep.
static std::vector<float> cpu_stencil2d_pass(const float* input, int n, int steps) {
    const size_t cells = (size_t)n * n;
    std::vector<float> ref(input, input + cells), next = ref;
    for (int t = 0; t < steps; ++t) {
        cpu_stencil2d(ref.data(), next.data(), n);
        ref.swap(next);
    }
    return ref;
}
