// Exercise the complete production pass. Small cases use independent double
// matrix products; large cases use integer rank-one factors with exact sums.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <utility>

#define main benchmark_main
#include "../../tier2/polybench_2mm/cuda/polybench_2mm.cu"
#undef main

enum Input { GENERAL, UPSTREAM, RANK_ONE };

static bool run_case(int N, float alpha, float beta, Input input) {
    const size_t count = static_cast<size_t>(N) * N;
    std::vector<float> A(count), B(count), C(count), D(count);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const size_t id = static_cast<size_t>(i) * N + j;
            if (input == UPSTREAM) {
                A[id] = static_cast<float>(i * j) / N;
                B[id] = static_cast<float>(i * (j + 1)) / N;
                C[id] = static_cast<float>(i * (j + 3)) / N;
                D[id] = static_cast<float>(i * (j + 2)) / N;
            } else if (input == RANK_ONE) {
                A[id] = (i % 11) * (j % 7);
                B[id] = (i % 7) * (j % 13 + 1);
                C[id] = (i % 5) * (j % 17 + 3);
                D[id] = (i % 3) * (j % 19 + 2);
            } else {
                A[id] = ((i * 7 + j * 3) % 17 - 8) / 8.0f;
                B[id] = ((i * 5 + j * 11) % 19 - 9) / 8.0f;
                C[id] = ((i * 13 + j * 2) % 23 - 11) / 8.0f;
                D[id] = ((i * 3 + j * 7) % 13 - 6) / 8.0f;
            }
        }
    }
    std::vector<double> want_tmp(count, 0.0), want_D(count, 0.0);
    if (input == RANK_ONE) {
        double ab_factor = 0.0, bc_factor = 0.0;
        for (int k = 0; k < N; ++k) {
            ab_factor += (k % 7) * (k % 7);
            bc_factor += (k % 13 + 1) * (k % 5);
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const size_t id = static_cast<size_t>(i) * N + j;
                want_tmp[id] = alpha * static_cast<double>(i % 11) * ab_factor * (j % 13 + 1);
                want_D[id] = alpha * static_cast<double>(i % 11) * ab_factor * bc_factor * (j % 17 + 3)
                             + beta * static_cast<double>(D[id]);
            }
        }
    } else {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k)
                    want_tmp[static_cast<size_t>(i) * N + j] +=
                        alpha * static_cast<double>(A[static_cast<size_t>(i) * N + k]) * B[static_cast<size_t>(k) * N + j];
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const size_t id = static_cast<size_t>(i) * N + j;
                want_D[id] = beta * static_cast<double>(D[id]);
                for (int k = 0; k < N; ++k)
                    want_D[id] += want_tmp[static_cast<size_t>(i) * N + k] * C[static_cast<size_t>(k) * N + j];
            }
        }
    }

    const float sentinel = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> tmp(count + 2, sentinel), output(count + 2, sentinel);
    std::copy(D.begin(), D.end(), output.begin() + 1);
    float *a, *b, *c, *d, *t;
    const size_t bytes = count * sizeof(float), guarded_bytes = (count + 2) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&a, bytes)); CUDA_CHECK(cudaMalloc(&b, bytes)); CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMalloc(&d, guarded_bytes)); CUDA_CHECK(cudaMalloc(&t, guarded_bytes));
    CUDA_CHECK(cudaMemcpy(a, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c, C.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d, output.data(), guarded_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t, tmp.data(), guarded_bytes, cudaMemcpyHostToDevice));
    run_2mm_pass(a, b, c, d + 1, t + 1, N, alpha, beta);
    CUDA_CHECK(cudaMemcpy(output.data(), d, guarded_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tmp.data(), t, guarded_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(t));
    if (!std::isnan(tmp.front()) || !std::isnan(tmp.back()) ||
        !std::isnan(output.front()) || !std::isnan(output.back())) {
        fprintf(stderr, "FAIL: output guard overwritten\n");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        for (int stage = 0; stage < 2; ++stage) {
            const double actual = stage ? output[i + 1] : tmp[i + 1];
            const double expected = stage ? want_D[i] : want_tmp[i];
            // Upstream uses a 0.05% difference threshold. Include an absolute
            // floor for signed cancellation/zero controls, and reject NaNs.
            if (!std::isfinite(actual) || !std::isfinite(expected) ||
                std::fabs(actual - expected) > 5e-4 * std::max(1.0, std::fabs(expected))) {
                fprintf(stderr, "FAIL: N=%d input=%d alpha=%g beta=%g stage=%d element=%zu got=%.12g want=%.12g\n",
                        N, input, alpha, beta, stage + 1, i, actual, expected);
                return false;
            }
        }
    }
    printf("PASS: N=%d input=%d alpha=%g beta=%g, all %zu intermediate/final elements and guards\n",
           N, input, alpha, beta, count);
    return true;
}

int main(int argc, char **argv) {
    setenv("BENCH_PARAM_vector_add_warmup", "false", 1);
    if (argc == 3 && strcmp(argv[1], "--rank-one") == 0) {
        const int n = atoi(argv[2]);
        if (n <= 0 || n > 4096) return 1;
        return run_case(n, 1.5f, 1.2f, RANK_ONE) ? 0 : 1;
    }
    if (argc != 1) return 1;
    const std::pair<float, float> cases[] = {
        {1.5f, 1.2f}, {0.0f, -0.75f}, {2.25f, 0.0f}, {-0.5f, 1.25f}
    };
    for (int n : {1, 2, 15, 16, 17, 31, 32, 33, 64, 65, 96, 128}) {
        for (const auto &coefficients : cases) {
            if (!run_case(n, coefficients.first, coefficients.second, GENERAL)) return 1;
        }
    }
    for (int n : {2, 15, 16, 17, 64, 96})
        if (!run_case(n, 32412.0f, 2123.0f, UPSTREAM)) return 1;
    for (int n : {64, 1024, 4096})
        if (!run_case(n, 1.5f, 1.2f, RANK_ONE)) return 1;
    return 0;
}
