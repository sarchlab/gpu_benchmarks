#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Complete independent host reference for valid grids (dimensions >= 3).
// Keep the qualified double arithmetic and upstream float coefficients.
static std::vector<double> cpu_stencil(const float* input, int nx, int ny, int nz,
                                       int steps) {
    const size_t cells = (size_t)nx * ny * nz;
    const auto index = [=](int x, int y, int z) {
        return ((size_t)z * ny + y) * nx + x;
    };
    const double center = (float)(1.0f / 6.0f);
    const double neighbor = (float)(1.0f / 6.0f / 6.0f);
    std::vector<double> ref(input, input + cells), next = ref;
    for (int t = 0; t < steps; ++t) {
        for (int z = 1; z < nz - 1; ++z)
            for (int y = 1; y < ny - 1; ++y)
                for (int x = 1; x < nx - 1; ++x) {
                    double sum = ref[index(x+1,y,z)] + ref[index(x-1,y,z)]
                               + ref[index(x,y+1,z)] + ref[index(x,y-1,z)]
                               + ref[index(x,y,z+1)] + ref[index(x,y,z-1)];
                    next[index(x,y,z)] = neighbor * sum - center * ref[index(x,y,z)];
                }
        ref.swap(next);
    }
    return ref;
}

static bool stencil_value_matches(float actual, double expected, bool boundary) {
    const double error = std::fabs((double)actual - expected);
    const double tolerance = boundary ? 0.0 : std::max(1e-7, 2e-5 * std::fabs(expected));
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::isfinite(error) && error <= tolerance;
}
