#pragma once
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <exception>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kmeans_reference {
// Acceptance recipe: float subtraction, one correctly rounded float FMA per
// feature, first minimum on ties, then serial float centroid accumulation.
inline bool value_matches(float actual, float expected) {
    return std::isfinite(actual) && std::isfinite(expected)
        && std::memcmp(&actual, &expected, sizeof(float)) == 0;
}
inline size_t history_elements(int n, int iterations) {
    if (n < 1 || iterations < 1
        || static_cast<size_t>(iterations) > std::numeric_limits<size_t>::max() / sizeof(int) / static_cast<size_t>(n))
        throw std::length_error("k-means membership history is not representable");
    return static_cast<size_t>(n) * iterations;
}

struct Result {
    std::vector<int> membership;
    std::vector<float> centers;
};

inline Result trajectory(const float* points, const float* initial, int n, int d, int k,
                  int iterations, unsigned workers = std::thread::hardware_concurrency()) {
    Result result;
    result.membership.resize(history_elements(n, iterations));
    result.centers.assign(initial, initial + static_cast<size_t>(k) * d);
    workers = std::max(1u, std::min({workers, 16u, static_cast<unsigned>(n)}));
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    std::vector<int> counts(k);
    std::vector<float> sums(static_cast<size_t>(k) * d);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int* labels = result.membership.data() + static_cast<size_t>(iteration) * n;
        auto assign = [&](unsigned worker) {
            const size_t first = static_cast<size_t>(n) * worker / workers;
            const size_t end = static_cast<size_t>(n) * (worker + 1) / workers;
            for (size_t p = first; p < end; ++p) {
                float best = FLT_MAX;
                int nearest = -1;
                for (int c = 0; c < k; ++c) {
                    float distance = 0.0f;
                    for (int f = 0; f < d; ++f) {
                        const float delta = points[p * d + f] - result.centers[c * d + f];
                        distance = std::fma(delta, delta, distance);
                    }
                    if (distance < best) { best = distance; nearest = c; }
                }
                labels[p] = nearest;
            }
        };
        for (unsigned worker = 1; worker < workers; ++worker) {
            try { threads.emplace_back(assign, worker); }
            catch (const std::exception&) { assign(worker); }
        }
        assign(0);
        for (auto& thread : threads) thread.join();
        threads.clear();
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0f);
        for (int p = 0; p < n; ++p) {
            const int cluster = labels[p];
            if (cluster < 0 || cluster >= k)
                throw std::runtime_error("k-means CPU reference produced an invalid membership");
            ++counts[cluster];
            for (int f = 0; f < d; ++f)
                sums[cluster * d + f] += points[static_cast<size_t>(p) * d + f];
        }
        for (int c = 0; c < k; ++c)
            if (counts[c])
                for (int f = 0; f < d; ++f)
                    result.centers[c * d + f] = sums[c * d + f] / counts[c];
    }
    return result;
}
}
