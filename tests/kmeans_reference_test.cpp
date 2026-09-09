#include "../tier2/rodinia_kmeans/cpu_reference.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <climits>
#include "kmeans_fma_golden.h"

static int cases = 0;
void check(const std::vector<float>& points, int d, int k, int iterations,
           const std::vector<int>& expected, const std::vector<float>& centers) {
    const int n = points.size() / d;
    const auto original = points;
    for (unsigned workers : {1u, 4u, 16u}) {
        auto result = kmeans_reference::trajectory(points.data(), points.data(),
                                                         n, d, k, iterations, workers);
        assert(points == original);
        assert(result.centers == centers);
        for (int t = 0; t < iterations; ++t)
            for (int p = 0; p < n; ++p)
                assert(result.membership[static_cast<size_t>(t) * n + p] == expected[p]);
        ++cases;
    }
}
void suite() {
    for (int t : {1, 2, 10}) {
        check({-1, 1, 0, 2}, 1, 2, t, {0, 1, 0, 1}, {-.5f, 1.5f});
        check({0, 0, 0, 0, 10, 10, 8, 8}, 2, 3, t,
                     {0, 0, 2, 2}, {0, 0, 0, 0, 9, 9});
        check({2, 2, 2, 2, 10, 10, 8, 8}, 2, 3, t,
                     {0, 0, 2, 2}, {2, 2, 2, 2, 9, 9});
        for (int d : {1, 3, 7, 34}) for (int k : {1, 2, 3, 5})
            for (int n : {k, k + 1, 17, 65}) {
                std::vector<float> points(static_cast<size_t>(n) * d), centers(k * d);
                std::vector<int> labels(n);
                for (int p = 0; p < n; ++p) {
                    labels[p] = p % k;
                    for (int f = 0; f < d; ++f)
                        points[p * d + f] = (p % k) * 32 + (p / k) % 3 + f - 64;
                }
                for (int c = 0; c < k; ++c) for (int f = 0; f < d; ++f) {
                    long double sum = 0; int count = 0;
                    for (int p = c; p < n; p += k) {sum += points[p * d + f]; ++count;}
                    centers[c * d + f] = static_cast<float>(sum / count);
                }
                check(points, d, k, t, labels, centers);
            }
    }
}
int main() {
    suite();
    for (const auto& fixture : golden) {
        std::vector<float> points(fixture.points.size());
        for (size_t at=0; at<points.size(); ++at)
            std::memcpy(&points[at], &fixture.points[at], sizeof(float));
        const auto original=points;
        for (unsigned workers : {0u,1u,4u,16u,32u}) {
            const auto result=kmeans_reference::trajectory(points.data(),points.data(),
                fixture.n,fixture.d,fixture.k,fixture.t,workers);
            assert(result.membership==fixture.labels && points==original);
            for (size_t at=0; at<result.centers.size(); ++at)
                assert(std::memcmp(&result.centers[at],&fixture.centers[at],sizeof(float))==0);
            ++cases;
        }
    }
    for (float value : {0.0f,1.0f,-1.0f,100.0f}) {
        assert(kmeans_reference::value_matches(value,value));
        assert(!kmeans_reference::value_matches(std::nextafter(value,INFINITY),value));
        for (float bad : {NAN,INFINITY,-INFINITY}) {
            assert(!kmeans_reference::value_matches(bad,value));
            assert(!kmeans_reference::value_matches(value,bad));
            assert(!kmeans_reference::value_matches(bad,bad));
        }
    }
    assert(!kmeans_reference::value_matches(-0.0f,0.0f));
    assert(kmeans_reference::history_elements(4194304,10)==41943040);
    for (int bad : {0,-1,INT_MIN}) {
        bool rejected=false;
        try { kmeans_reference::history_elements(bad,10); }
        catch (const std::length_error&) {rejected=true;}
        assert(rejected);
    }
    std::printf("%d known-label/center cases passed, exact FMA and 1/4/16 workers\n", cases);
}
