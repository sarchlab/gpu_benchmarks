// Independent old-formula and known-answer checks for the full LU residual.
#include "../tier2/rodinia_lud/hip/lud_residual.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

static bool same(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(a)) == 0;
}

static double unblocked(const std::vector<float>& lu, const std::vector<float>& a,
                        int n, std::vector<double>& output) {
    double err = 0.0, norm_a = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            const int limit = i < j ? i : j;
            for (int k = 0; k <= limit; ++k) {
                const double l = k == i ? 1.0 : static_cast<double>(lu[i * n + k]);
                const double u = static_cast<double>(lu[k * n + j]);
                sum += l * u;
            }
            output[i * n + j] = sum;
            const double diff = static_cast<double>(a[i * n + j]) - sum;
            err += diff * diff;
            norm_a += static_cast<double>(a[i * n + j]) * static_cast<double>(a[i * n + j]);
        }
    }
    return std::sqrt(err / norm_a);
}

static bool check(const std::vector<float>& lu, const std::vector<float>& a, int n) {
    std::vector<double> expected(n * n);
    const double old_residual = unblocked(lu, a, n, expected);
    const double residual = lud_reference::relative_residual(lu.data(), a.data(), n);
    if (!same(old_residual, residual) || (old_residual < 1e-4) != (residual < 1e-4)) {
        std::fprintf(stderr, "Residual mismatch n=%d: %.17g / %.17g\n", n, old_residual, residual);
        return false;
    }
    // Guard both the output buffer and unused rows of the final partial block.
    const double guard = -123456789.25;
    std::vector<double> block(lud_reference::row_block * n + 2, guard);
    for (int first = 0; first < n; first += lud_reference::row_block) {
        std::fill(block.begin(), block.end(), guard);
        lud_reference::reconstruct_rows(lu.data(), n, first, block.data() + 1);
        const int used = std::min(n - first, lud_reference::row_block) * n;
        if (block.front() != guard) return false;
        for (std::size_t at = used + 1; at < block.size(); ++at)
            if (block[at] != guard) return false;
        for (int at = 0; at < used; ++at) {
            if (!same(expected[first * n + at], block[at + 1])) {
                std::fprintf(stderr, "Reconstruction mismatch n=%d row=%d offset=%d\n", n, first, at);
                return false;
            }
        }
    }
    return true;
}

int main() {
    std::mt19937 rng(42);
    int cases = 0;
    for (int n : {1,2,3,4,5,7,15,16,17,31,32,33,63,64,65,95,96,97,127,128,129,192,256}) {
        std::vector<float> lu(n * n), a(n * n);
        for (int pattern = 0; pattern < 7; ++pattern) {
            for (int at = 0; at < n * n; ++at) {
                lu[at] = (static_cast<int>(rng() % 2001) - 1000) * .001f;
                a[at] = (static_cast<int>(rng() % 2001) - 1000) * .001f;
                if (pattern == 1) lu[at] = a[at] = at / n == at % n ? 1.f : 0.f;
                if (pattern == 2) { lu[at] = at % 2 ? 0.f : -0.f; a[at] = 0.f; }
                if (pattern == 3) {
                    lu[at] = std::ldexp(static_cast<float>(static_cast<int>(rng() % 2049) - 1024),
                                        static_cast<int>(rng() % 251) - 140);
                    a[at] = lu[at];
                }
                if (pattern >= 4 && at == n * n / 2)
                    lu[at] = pattern == 4 ? std::numeric_limits<float>::quiet_NaN() :
                              pattern == 5 ? std::numeric_limits<float>::infinity() :
                                             -std::numeric_limits<float>::infinity();
            }
            if (!check(lu, a, n)) return 1;
            ++cases;
        }
    }
    const std::vector<float> lu = {5,6,7, 2,8,9, 3,4,10};
    const std::vector<float> a = {5,6,7, 10,20,23, 15,50,67};
    if (!check(lu, a, 3) || lud_reference::relative_residual(lu.data(), a.data(), 3) != 0.0) return 1;
    ++cases;
    for (float value : {1.00009f, 1.00011f}) {
        const std::vector<float> unit = {1.f}, original = {value};
        if (!check(unit, original, 1) ||
            (lud_reference::relative_residual(unit.data(), original.data(), 1) < 1e-4) != (value < 1.0001f)) return 1;
        ++cases;
    }
    for (int at : {0,1,3,4,8}) {
        for (float value : {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(), 100000.f}) {
            auto bad = lu;
            bad[at] = value;
            if (!check(bad, a, 3) || lud_reference::relative_residual(bad.data(), a.data(), 3) < 1e-4) return 1;
            ++cases;
        }
    }
    if (cases != 184) return 1;
    std::printf("%d full LU reconstruction/residual cases pass\n", cases);
    return 0;
}
