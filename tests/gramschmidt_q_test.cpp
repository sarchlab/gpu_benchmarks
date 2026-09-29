// The driver prepends the actual production Q orthogonality checker.
#include <cassert>
#include <cstring>
#include <limits>
#include <vector>

static size_t cases = 0;
static void check(const std::vector<float>& q, int m, int n, size_t expected_errors) {
    std::vector<float> guarded(q.size() + 2, 12345.0f);
    std::memcpy(guarded.data() + 1, q.data(), q.size() * sizeof(float));
    const auto before = guarded;
    fprintf(stderr, "GRAM_Q_CASE %zu %d %zu\n", cases, n, expected_errors);
    assert(check_orthogonality(guarded.data() + 1, m, n) == (expected_errors == 0));
    assert(std::memcmp(guarded.data(), before.data(), guarded.size() * sizeof(float)) == 0);
    ++cases;
}

int main() {
    // Columns of a Householder reflector are orthonormal by construction.
    // Dense nonzero rows exercise the full row extent, including the largest M.
    for (int m : {9,17,64,65,257,6144}) for (int n : {1,7,8,9}) {
        std::vector<float> q(m*n);
        for (int i=0; i<m; ++i) for (int j=0; j<n; ++j)
            q[i*n+j] = (i == j ? 1.0f : 0.0f) - 2.0f/m;
        check(q,m,n,0);
    }
    // Exact signed/permuted basis and normalized 64x64 Walsh-Hadamard columns.
    for (int m : {64,65,6144}) for (int n : {8,9,33,64}) {
        std::vector<float> basis(m*n), hadamard(m*n);
        for (int j=0; j<n; ++j) basis[(m-1-j)*n+j] = j%2 ? -1.0f : 1.0f;
        for (int i=0; i<64; ++i) for (int j=0; j<n; ++j) {
            unsigned bits = (unsigned)(i & j), parity = 0;
            while (bits) { parity ^= bits & 1u; bits >>= 1; }
            hadamard[i*n+j] = parity ? -0.125f : 0.125f;
        }
        check(basis,m,n,0); check(hadamard,m,n,0);
    }
    const int m=17, n=9;
    std::vector<float> q(m*n);
    for (int j=0; j<n; ++j) q[(m-1-j)*n+j] = 1.0f;
    // Every element, including the formerly unchecked ninth column and last row.
    for (size_t at=0; at<q.size(); ++at) {
        auto bad=q; bad[at]=4.0f;
        const int row=(int)at/n, column=(int)at%n;
        // One bad norm, plus two off-diagonals if this row contains another
        // column's unit entry. Rows above the basis have no unit entries.
        const size_t errors=1 + (row>=m-n && row!=m-1-column ? 2 : 0);
        check(bad,m,n,errors);
    }
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        auto bad=q; bad.back()=value; check(bad,m,n,2*n-1);
    }
    // Unit norms alone are insufficient: duplicate columns violate off-diagonals.
    auto duplicate=q;
    for (int i=0; i<m; ++i) duplicate[i*n+n-1]=duplicate[i*n];
    check(duplicate,m,n,2);
    check(std::vector<float>(m*n,0.0f),m,n,n);

    struct Fixture { float value; size_t errors; };
    // The off-diagonal dot is exactly epsilon. The equality case pins the
    // inclusive 0.01f bound; adjacent floats catch an accidental >= change.
    for (Fixture f : {Fixture{0.005f,0}, {std::nextafter(0.01f,0.0f),0},
                      {0.01f,0}, {std::nextafter(0.01f,1.0f),2}, {0.02f,2}})
        check({1.0f,f.value,0.0f,std::sqrt(1.0f-f.value*f.value)},2,2,f.errors);
    // Norm-squared errors on both sides of 1, comfortably inside/outside the
    // same bound. No off-diagonal error is present in these fixtures.
    for (Fixture f : {Fixture{-0.02f,1}, {-0.005f,0}, {0.005f,0}, {0.02f,1}})
        check({1.0f,0.0f,0.0f,std::sqrt(1.0f+f.value)},2,2,f.errors);
    printf("PASS: %zu independent Q-property fixtures, tolerance boundaries, immutable inputs and guards\n",cases);
}
