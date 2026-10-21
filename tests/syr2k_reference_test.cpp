#include <cstdio>
#include <limits>
#include "../tier2/polybench_syr2k/hip/cpu_reference.h"

int main() {
    int cases = 0;
    for (int n : {1, 2, 3, 15, 16, 17, 31, 32, 33, 64, 65})
        for (int m : {1, 2, 3, 15, 16, 17, 31, 32, 33, 65})
            for (float alpha : {1.5f, 0.0f, -0.5f, 2.25f})
                for (float beta : {1.2f, 0.0f, -0.75f}) {
                    std::vector<float> a((size_t)n*m), b(a.size()), c((size_t)n*n);
                    for (size_t i = 0; i < a.size(); ++i) {
                        a[i] = float(int(i*7%17)-8)/8;
                        b[i] = float(int(i*11%19)-9)/8;
                    }
                    for (size_t i = 0; i < c.size(); ++i)
                        c[i] = float(int(i*3%13)-6)/8;
                    const auto original_a = a, original_b = b, original_c = c;
                    const auto actual = cpu_syr2k(a.data(), b.data(), c.data(), n, m, alpha, beta);
                    if (actual.size() != c.size() || a != original_a ||
                        b != original_b || c != original_c) return 1;
                    // Scalar long-double oracle: two separate row dot products,
                    // with no transpose or blocked layout from the implementation.
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n; ++j) {
                            long double ab = 0, ba = 0;
                            for (int k = 0; k < m; ++k) {
                                ab += (long double)a[(size_t)i*m+k]*b[(size_t)j*m+k];
                                ba += (long double)b[(size_t)i*m+k]*a[(size_t)j*m+k];
                            }
                            const size_t q = (size_t)i*n+j;
                            const long double expected = alpha*(ab+ba)+(long double)beta*c[q];
                            if (std::fabs((long double)actual[q]-expected) >
                                1e-12L*std::max(1.0L,std::fabs(expected))) {
                                std::fprintf(stderr, "Reference mismatch N=%d M=%d at %zu\n", n,m,q);
                                return 1;
                            }
                        }
                    ++cases;
                }
    // The last inner coordinate contributes to two distinct off-diagonal
    // outputs; the nonsymmetric initial C must be retained independently.
    std::vector<float> a(15,0), b(15,0), c(9);
    a[4]=2; b[14]=3;
    for (size_t i=0;i<c.size();++i) c[i]=float(i+1)/8;
    auto impulse=cpu_syr2k(a.data(),b.data(),c.data(),3,5,2.0f,-0.5f);
    for (size_t q=0;q<c.size();++q)
        if (impulse[q] != ((q==2 || q==6)?12.0:0.0)-0.5*c[q]) return 1;
    if (!syr2k_value_matches(0,0) || !syr2k_value_matches(4e-4f,0) ||
        syr2k_value_matches(6e-4f,0) || !syr2k_value_matches(100.04f,100) ||
        syr2k_value_matches(100.06f,100) || !syr2k_value_matches(-100.04f,-100) ||
        syr2k_value_matches(-100.06f,-100)) return 1;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()})
        if (syr2k_value_matches(bad,0) || syr2k_value_matches(0,bad) ||
            syr2k_value_matches(bad,bad)) return 1;
    std::printf("PASS: %d complete rectangular references, two-product tail impulse, input preservation and finite/tolerance checks\n",cases);
    return 0;
}
