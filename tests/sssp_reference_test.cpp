// The driver prepends the actual production CPU Bellman-Ford reference.
#include <algorithm>
#include <cassert>

int main() {
    size_t cases = 0;
    // Every directed four-vertex graph, unit/varying positive weights, all sources.
    for (int mask = 0; mask < (1 << 12); ++mask) for (int mode : {0,1}) {
        std::vector<int> rows{0}, cols, weights;
        int oracle[16];
        for (int u = 0; u < 4; ++u) for (int v = 0; v < 4; ++v)
            oracle[u*4+v] = u == v ? 0 : INT_MAX;
        int bit = 0;
        for (int u = 0; u < 4; ++u) {
            for (int v = 0; v < 4; ++v) {
                if (u == v) continue;
                if (mask & (1 << bit)) {
                    const int w = mode ? 1 + (u*17 + v*13) % 100 : 1;
                    cols.push_back(v); weights.push_back(w); oracle[u*4+v] = w;
                }
                ++bit;
            }
            rows.push_back((int)cols.size());
        }
        for (int k = 0; k < 4; ++k) for (int u = 0; u < 4; ++u) for (int v = 0; v < 4; ++v)
            if (oracle[u*4+k] != INT_MAX && oracle[k*4+v] != INT_MAX)
                oracle[u*4+v] = std::min(oracle[u*4+v], oracle[u*4+k] + oracle[k*4+v]);
        for (int source = 0; source < 4; ++source) {
            std::vector<int> distances;
            const int iterations = bellman_ford_cpu(4,rows,cols,weights,distances,source);
            assert(iterations >= 1 && iterations <= 4 && distances.size() == 4);
            for (int v = 0; v < 4; ++v) assert(distances[v] == oracle[source*4+v]);
            ++cases;
        }
    }
    // Positive self-loop, parallel edges, weighted chain and unreachable component.
    const std::vector<int> rows{0,3,4,5,5,6,6}, cols{0,1,1,2,3,5}, weights{7,9,2,3,4,1};
    std::vector<int> distances;
    bellman_ford_cpu(6,rows,cols,weights,distances,0);
    assert((distances == std::vector<int>{0,2,5,9,INT_MAX,INT_MAX}));
    printf("PASS: %zu exhaustive CPU/Floyd-Warshall cases and weighted parallel/self/unreachable fixture\n",cases);
}
