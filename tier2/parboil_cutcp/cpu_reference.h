#ifndef PARBOIL_CUTCP_CPU_REFERENCE_H
#define PARBOIL_CUTCP_CPU_REFERENCE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cutcp_reference {

inline std::size_t lattice_size(int n) {
    if (n <= 0) throw std::invalid_argument("nonpositive reference lattice dimension");
    const std::size_t side = static_cast<std::size_t>(n);
    const std::size_t limit = std::vector<float>().max_size();
    if (side > limit / side || side * side > limit / side)
        throw std::length_error("reference lattice is too large");
    return side * side * side;
}

// Conservative boxes skip only points outside an atom's cutoff. Clamp in
// double before converting to int; the one-cell margin also covers rounding
// at the cutoff. The original float distance test remains authoritative.
inline int lower_bound(float coordinate, float cutoff, float h, int n) {
    const double bound = std::floor((double(coordinate) - cutoff) / h) - 1;
    return static_cast<int>(std::max(0.0, std::min(double(n), bound)));
}
inline int upper_bound(float coordinate, float cutoff, float h, int n) {
    const double bound = std::ceil((double(coordinate) + cutoff) / h) + 1;
    return static_cast<int>(std::max(-1.0, std::min(double(n - 1), bound)));
}

// Atom has x/y/z/w fields. The caller supplies the actual nonempty bins in
// their original flattened-bin/insertion order. Each point sums in that same
// order with the former pointwise CPU expression; no GPU work is replayed.
template <class Atom>
std::vector<float> potential(const std::vector<Atom>& atoms, int n,
                             float h, float cutoff, unsigned requested = 0) {
    const std::size_t count = lattice_size(n);
    const float cutoff2 = cutoff * cutoff;
    const float inv_cutoff2 = 1.f / cutoff2;
    if (!std::isfinite(h) || h <= 0.f || !std::isfinite(cutoff) || cutoff <= 0.f ||
        !std::isfinite(cutoff2) || !std::isfinite(inv_cutoff2) ||
        !std::isfinite((n - 1) * h))
        throw std::invalid_argument("invalid reference spacing or cutoff");
    for (const Atom& a : atoms)
        if (!std::isfinite(a.x) || !std::isfinite(a.y) ||
            !std::isfinite(a.z) || !std::isfinite(a.w))
            throw std::invalid_argument("nonfinite reference atom");

    std::vector<float> result(count, 0.f);
    unsigned workers = requested ? requested : std::thread::hardware_concurrency();
    workers = std::max(1u, std::min(16u, std::min(workers, unsigned(n))));
    auto part = [&](unsigned worker) {
        const int first_z = std::size_t(n) * worker / workers;
        const int end_z = std::size_t(n) * (worker + 1) / workers;
        for (const Atom& a : atoms) {
            const int x0 = lower_bound(a.x, cutoff, h, n);
            const int x1 = upper_bound(a.x, cutoff, h, n);
            const int y0 = lower_bound(a.y, cutoff, h, n);
            const int y1 = upper_bound(a.y, cutoff, h, n);
            const int z0 = std::max(first_z, lower_bound(a.z, cutoff, h, n));
            const int z1 = std::min(end_z - 1, upper_bound(a.z, cutoff, h, n));
            for (int k = z0; k <= z1; ++k)
                for (int j = y0; j <= y1; ++j)
                    for (int i = x0; i <= x1; ++i) {
                        const float x = i * h, y = j * h, z = k * h;
                        const float dx = a.x - x, dy = a.y - y, dz = a.z - z;
                        const float r2 = dx*dx + dy*dy + dz*dz;
                        if (r2 < cutoff2) {
                            const float s = 1.f - r2 * inv_cutoff2;
                            result[(std::size_t(k) * n + j) * n + i] +=
                                a.w * (1.f / sqrtf(r2)) * s * s;
                        }
                    }
        }
    };
    // Slabs are disjoint. If creation fails after some workers started, the
    // caller completes the remaining slabs and joins every started worker.
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    unsigned worker = 1;
    try {
        for (; worker < workers; ++worker) threads.emplace_back(part, worker);
    } catch (const std::exception&) {
        for (; worker < workers; ++worker) part(worker);
    }
    part(0);
    for (auto& thread : threads) thread.join();
    return result;
}

} // namespace cutcp_reference
#endif
