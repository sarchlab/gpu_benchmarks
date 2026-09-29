// The driver prepends the actual production reduction and NAS constants.
#include <cassert>
#include <limits>
#include <utility>
#include <vector>

int main() {
    int cases = 0;
    auto run = [&](long long nn, int threads, std::vector<double> q,
                   std::vector<double> x, std::vector<double> y, size_t expected) {
        const int blocks = (int)x.size();
        assert(y.size() == x.size() && q.size() == x.size() * NQ);
        double out[NQ + 2], sx, sy;
        out[0] = out[NQ + 1] = 12345.0;
        const size_t errors = reduce_and_check_partials(q.data(), x.data(), y.data(),
                nn, threads, blocks, out + 1, sx, sy);
        assert(errors == expected && out[0] == 12345.0 && out[NQ + 1] == 12345.0);
        // The original reduction order must remain unchanged, including failures.
        double old_q[NQ] = {}, old_sx = 0, old_sy = 0;
        for (int b = 0; b < blocks; ++b) {
            for (int i = 0; i < NQ; ++i) old_q[i] += q[(size_t)b * NQ + i];
            old_sx += x[b]; old_sy += y[b];
        }
        auto same = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
        for (int i = 0; i < NQ; ++i) assert(same(out[i + 1], old_q[i]));
        assert(same(sx, old_sx) && same(sy, old_sy));
        ++cases;
    };
    // Full and partial blocks, zeros, exact inclusive budgets, maximum admitted M.
    for (const auto& geometry : std::vector<std::pair<long long,int>>{
            {2,1}, {2,3}, {7,3}, {256,128}, {256,512}, {1LL<<30,2147483647}}) {
        const long long nn = geometry.first;
        const int threads = geometry.second, blocks = (int)((nn + threads - 1) / threads);
        std::vector<double> q((size_t)blocks * NQ), x(blocks, 0.5), y(blocks, -0.5);
        run(nn, threads, q, x, y, 0);
        for (int b = 0; b < blocks; ++b) {
            const long long remaining = nn - (long long)b * threads;
            q[(size_t)b * NQ + 9] = (double)(remaining < threads ? remaining : threads) * NK;
        }
        run(nn, threads, q, x, y, 0);
    }
    for (int block : {0,2}) {
        const double budget = (block == 2 ? 1.0 : 3.0) * NK;
        for (int bin : {0,9}) {
            for (double value : {double(NAN), double(INFINITY), -double(INFINITY), -1.0, 0.5, budget + 1}) {
                std::vector<double> q(30), x(3,1.0), y(3,-1.0);
                q[block * NQ + bin] = value;
                run(7,3,q,x,y,1);
            }
        }
        std::vector<double> q(30), x(3,1.0), y(3,-1.0);
        q[block * NQ] = budget / 2 + 1;
        q[block * NQ + 9] = budget / 2;
        run(7,3,q,x,y,1); // each bin legal, block total over budget
        for (double value : {double(NAN), double(INFINITY), -double(INFINITY)}) {
            x[block] = value; run(7,3,std::vector<double>(30),x,y,2); x[block] = 1;
            y[block] = value; run(7,3,std::vector<double>(30),x,y,2); y[block] = -1;
        }
    }
    run(7,3,std::vector<double>(30,-1),std::vector<double>(3),std::vector<double>(3),30);
    // Finite partials alone are insufficient: final sums may overflow.
    const double huge = std::numeric_limits<double>::max();
    run(2,1,std::vector<double>(20),{huge,huge},{0,0},1);
    run(2,1,std::vector<double>(20),{0,0},{-huge,-huge},1);
    printf("PASS: %d production EP sanity/reduction fixtures\n", cases);
}
