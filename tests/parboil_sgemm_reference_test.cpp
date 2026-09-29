#include <cstdio>
#include <cstring>
#include "../tier2/parboil_sgemm/cpu_reference.h"

static void original(const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * s + beta * C[i * N + j];
        }
    }
}

int main() {
    size_t cases = 0;
    for (int m : {1, 3, 4, 5, 17, 33}) for (int n : {1, 7, 31, 32, 33})
        for (int k : {1, 2, 17, 33, 65}) for (unsigned workers : {1u, 2u, 8u, 16u}) {
            std::vector<float> a((size_t)m * k), b((size_t)k * n), c((size_t)m * n + 2);
            for (size_t q = 0; q < a.size(); ++q) a[q] = float(int(q * 7 % 31) - 15) * .137f;
            for (size_t q = 0; q < b.size(); ++q) b[q] = float(int(q * 11 % 29) - 14) * .139f;
            for (size_t q = 0; q < c.size(); ++q) c[q] = float(int(q * 13 % 23) - 11) * .141f;
            c.front() = 12345.f; c.back() = -12345.f;
            const auto aa = a, bb = b;
            auto expected = c;
            original(a.data(), b.data(), expected.data() + 1, m, n, k, 1.37f, -.29f);
            gemm_cpu(a.data(), b.data(), c.data() + 1, m, n, k, 1.37f, -.29f, workers);
            if (a != aa || b != bb || std::memcmp(c.data(), expected.data(), c.size() * sizeof(float)))
                return 1;
            ++cases;
        }
    // Exercise automatic worker selection at both sides of the output threshold.
    for (int m : {511, 512}) {
        const int n = 512, k = 2;
        std::vector<float> a((size_t)m * k), b((size_t)k * n), c((size_t)m * n);
        for (size_t i = 0; i < a.size(); ++i) a[i] = float(i % 7) * .137f;
        for (size_t i = 0; i < b.size(); ++i) b[i] = float(i % 11) * .139f;
        for (size_t i = 0; i < c.size(); ++i) c[i] = float(i % 13) * .141f;
        auto expected = c;
        original(a.data(), b.data(), expected.data(), m, n, k, 1.5f, 1.2f);
        gemm_cpu(a.data(), b.data(), c.data(), m, n, k, 1.5f, 1.2f);
        if (std::memcmp(c.data(), expected.data(), c.size() * sizeof(float))) return 6;
        ++cases;
    }
    // Independent rectangular known answer includes nonzero initial C and beta.
    const float a[] = {1, 2, 3, 4, 5, 6}, b[] = {7, 8, 9, 10, 11, 12};
    float c[] = {1, 2, 3, 4};
    const float expected[] = {114, 124, 272, 300};
    gemm_cpu(a, b, c, 2, 2, 3, 2.f, -2.f);
    if (std::memcmp(c, expected, sizeof(c))) return 2;
    // Zero A must leave exactly beta*C; identity multiplication preserves B.
    float zero[] = {0, 0, 0, 0}, id[] = {1, 0, 0, 1}, value[] = {2, -3, 4, -5};
    float z[] = {1, 2, 3, 4};
    gemm_cpu(zero, value, z, 2, 2, 2, 1.f, -2.f);
    for (int i = 0; i < 4; ++i) if (z[i] != -2.f * (i + 1)) return 3;
    gemm_cpu(id, value, z, 2, 2, 2, 1.f, 0.f);
    if (std::memcmp(z, value, sizeof(z))) return 4;
    for (int axis = 0; axis < 3; ++axis) for (int bad : {0, -1}) {
        int dims[] = {2, 2, 2}; dims[axis] = bad;
        try { gemm_cpu(id, value, z, dims[0], dims[1], dims[2], 1.f, 0.f); return 5; }
        catch (const std::invalid_argument&) {}
    }
    std::printf("PASS %zu complete bitwise references, rectangular known answer, zero/identity, input/guard preservation and invalid dimensions\n", cases);
}
