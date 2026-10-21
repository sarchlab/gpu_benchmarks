#pragma once
#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

// Preserve the original FP32 increasing-k recipe while sharing B cache lines
// across outputs. Workers own disjoint row tiles; C starts as pristine input.
static void gemm_cpu(const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha, float beta,
                     unsigned requested_workers = 0) {
    if (M <= 0 || N <= 0 || K <= 0)
        throw std::invalid_argument("reference dimensions must be positive");
    unsigned wanted = requested_workers;
    if (!wanted)
        wanted = (size_t)M * N < 512u * 512u ? 1u :
                 std::max(1u, std::thread::hardware_concurrency());
    const unsigned workers = std::max(1u, std::min({8u, wanted, ((unsigned)M + 3) / 4}));
    auto compute = [&](unsigned worker) {
        for (size_t i = worker * 4; i < (size_t)M; i += workers * 4) {
            for (int j = 0; j < N; j += 32) {
                const int nr = std::min(4, M - (int)i), nc = std::min(32, N - j);
                float acc[4][32] = {};
                for (int k = 0; k < K; ++k) {
                    for (int r = 0; r < nr; ++r) {
                        const float av = A[(i + r) * K + k];
                        for (int c = 0; c < nc; ++c)
                            acc[r][c] += av * B[(size_t)k * N + j + c];
                    }
                }
                for (int r = 0; r < nr; ++r) {
                    for (int c = 0; c < nc; ++c) {
                        const size_t q = (i + r) * N + j + c;
                        C[q] = alpha * acc[r][c] + beta * C[q];
                    }
                }
            }
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    unsigned worker = 1;
    try {
        for (; worker < workers; ++worker)
            threads.emplace_back(compute, worker);
    } catch (const std::exception&) {
        // Already started workers retain their partitions; complete the rest
        // on the caller and join everything before returning.
        for (; worker < workers; ++worker) compute(worker);
    }
    compute(0);
    for (auto& thread : threads) thread.join();
}
