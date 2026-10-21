#include "../tier2/rodinia_bfs/hip/cpu_reference.h"
#include <algorithm>
#include <cstdio>

// Independent all-pairs Floyd-Warshall oracle; no frontier/queue logic.
static bool check(const std::vector<std::vector<int>>& edges) {
    const int n = static_cast<int>(edges.size()), infinity = n + 1;
    std::vector<int> row(n+1), col, paths(n*n, infinity);
    for (int u = 0; u < n; ++u) {
        paths[u*n+u] = 0;
        for (int v : edges[u]) {
            col.push_back(v);
            paths[u*n+v] = std::min(paths[u*n+v], 1);
        }
        row[u+1] = static_cast<int>(col.size());
    }
    for (int k = 0; k < n; ++k)
        for (int u = 0; u < n; ++u)
            for (int v = 0; v < n; ++v)
                paths[u*n+v] = std::min(paths[u*n+v], paths[u*n+k] + paths[k*n+v]);
    BfsWork work;
    const auto actual = bfs_reference(row.data(), col.data(), n, &work);
    BfsWork expected_work;
    for (int u=0;u<n;++u) if (paths[u]!=infinity) {
        expected_work.levels=std::max(expected_work.levels,static_cast<unsigned long long>(paths[u])+1);
        if (u) ++expected_work.promotion_atomics;
        for (int v : edges[u]) {
            ++expected_work.frontier_edges;
            if (paths[v]==paths[u]+1) ++expected_work.discovery_atomics;
        }
    }
    if (work.levels!=expected_work.levels || work.frontier_edges!=expected_work.frontier_edges
        || work.discovery_atomics!=expected_work.discovery_atomics
        || work.promotion_atomics!=expected_work.promotion_atomics) {
        std::fprintf(stderr,"Work-count mismatch n=%d\n",n);return false;
    }
    if (actual.size() != static_cast<size_t>(n)) return false;
    for (int v = 0; v < n; ++v) {
        const int expected = paths[v] == infinity ? -1 : paths[v];
        if (actual[v] != expected) {
            std::fprintf(stderr, "Distance mismatch n=%d vertex=%d actual=%d expected=%d\n",
                         n, v, actual[v], expected);
            return false;
        }
    }
    return true;
}

int main() {
    unsigned cases = 0;
    // Every directed graph through four vertices, including every self-loop.
    for (int n = 1; n <= 4; ++n) {
        for (unsigned mask = 0; mask < (1u << (n*n)); ++mask) {
            std::vector<std::vector<int>> edges(n);
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    if (mask & (1u << (u*n+v))) edges[u].push_back(v);
            if (!check(edges)) return 1;
            ++cases;
        }
    }
    // Duplicate destinations, deep chains, reverse edges, disconnected cycles,
    // and multiple competing path lengths. Vary adjacency order independently.
    for (int n : {5,9,17,64,129}) {
        for (int pattern = 0; pattern < 5; ++pattern) {
            std::vector<std::vector<int>> edges(n);
            for (int u = 0; u < n; ++u) {
                edges[u].push_back(u);
                if (pattern == 0 && u+1 < n) edges[u].push_back(u+1);
                if (pattern == 1 && u) edges[u].push_back(u-1);
                if (pattern == 2 && u+1 < n/2) edges[u].push_back(u+1);
                if (pattern == 2 && u >= n/2) edges[u].push_back(u+1 < n ? u+1 : n/2);
                if (pattern == 3) for (int v = 0; v < n; ++v) edges[u].push_back(v);
                if (pattern == 4) {
                    if (u+1 < n) edges[u].push_back(u+1);
                    edges[u].push_back((u*7+3)%n);
                }
                const auto duplicate = edges[u];
                edges[u].insert(edges[u].end(), duplicate.begin(), duplicate.end());
                if (u%2) std::reverse(edges[u].begin(), edges[u].end());
            }
            if (!check(edges)) return 1;
            ++cases;
        }
    }
    if (cases != 66091) return 1;
    std::printf("%u directed-graph distance and logical-work comparisons pass (66066 exhaustive,25 structured)\n", cases);
    return 0;
}
