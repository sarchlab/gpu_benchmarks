// The driver prepends the production synchronous frontier CPU reference.
#include <algorithm>
#include <cassert>

// Deliberately dense snapshot recurrence: visit every reachable source on
// every round, independently of the production frontier queue and work formula.
static int dense_rounds(int n, const std::vector<int>& row, const std::vector<int>& col,
                        const std::vector<int>& weights, int source,
                        std::vector<int>& dist, unsigned long long& edges) {
    dist.assign(n, INT_MAX); dist[source] = 0; edges = 0;
    int rounds = 0;
    for (; rounds < n;) {
        const std::vector<int> previous = dist;
        bool changed = false;
        for (int u = 0; u < n; ++u) if (previous[u] != INT_MAX) {
            for (int e = row[u]; e < row[u+1]; ++e) {
                ++edges;
                const int v = col[e], candidate = previous[u] + weights[e];
                if (candidate < dist[v]) { dist[v] = candidate; changed = true; }
            }
        }
        ++rounds;
        if (!changed) break;
    }
    return rounds;
}

static void check_rounds(int n, const std::vector<int>& row, const std::vector<int>& col,
                         const std::vector<int>& weights, int source) {
    std::vector<int> got, want;
    SsspWork work;
    unsigned long long edges;
    int rounds = bellman_ford_cpu(n,row,col,weights,got,source,&work);
    int expected = dense_rounds(n,row,col,weights,source,want,edges);
    assert(got == want && rounds == expected && work.edge_relaxations == edges);
}

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
                    const int w = mode ? 1 + (u*17 + v*31 + u*v*29) % 100 : 1;
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
            check_rounds(4,rows,cols,weights,source);
            ++cases;
        }
    }
    // Positive self-loop, parallel edges, weighted chain and unreachable component.
    const std::vector<int> rows{0,3,4,5,5,6,6}, cols{0,1,1,2,3,5}, weights{7,9,2,3,4,1};
    std::vector<int> distances;
    bellman_ford_cpu(6,rows,cols,weights,distances,0);
    assert((distances == std::vector<int>{0,2,5,9,INT_MAX,INT_MAX}));
    check_rounds(6,rows,cols,weights,0);
    // Competing weighted paths improve an already-active source. Reading
    // live distances instead of its snapshot would propagate one round early.
    check_rounds(5,{0,2,3,4,5,5},{1,2,2,3,4},{1,10,1,1,1},0);
    // A path requires one round per edge plus the final no-change round;
    // forward and reverse vertex numbering must give identical work.
    for (int n : {1,2,31,257}) for (bool reverse : {false,true}) {
        std::vector<int> row{0}, col, weight;
        for (int u=0;u<n;++u) {
            int v=reverse ? u-1 : u+1;
            if(v>=0 && v<n) { col.push_back(v); weight.push_back(1); }
            row.push_back((int)col.size());
        }
        check_rounds(n,row,col,weight,reverse ? n-1 : 0);
        SsspWork work;
        int rounds=bellman_ford_cpu(n,row,col,weight,distances,reverse ? n-1 : 0,&work);
        assert(rounds==n && work.edge_relaxations==(unsigned long long)n*(n-1)/2+(n-1));
    }
    printf("PASS: %zu exhaustive CPU/Floyd-Warshall cases and weighted parallel/self/unreachable fixture\n",cases);
}
