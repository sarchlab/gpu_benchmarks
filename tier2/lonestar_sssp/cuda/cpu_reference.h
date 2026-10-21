#pragma once
#include <algorithm>
#include <climits>
#include <vector>

struct SsspWork {
    unsigned long long edge_relaxations = 0;
};

// Synchronous CPU reference: unchanged sources need no repeated propagation.
// Snapshot all distances before each round, then scan active sources in CSR
// order for locality. Count the GPU's full reachable-edge workload separately,
// including the final no-change round. Weights are positive and all generated
// manifest distances are representable integers.
static int bellman_ford_cpu(int N, const std::vector<int>& row_offsets,
                             const std::vector<int>& col_indices,
                             const std::vector<int>& weights,
                             std::vector<int>& dist, int source,
                             SsspWork* work = nullptr) {
    if (work) *work = SsspWork{};
    dist.assign(N, INT_MAX);
    dist[source] = 0;
    std::vector<int> previous(N);
    std::vector<unsigned char> active(N, 0), next(N, 0);
    active[source] = 1;
    unsigned long long reachable_edges = row_offsets[source + 1] - row_offsets[source];
    int rounds = 0;
    for (; rounds < N;) {
        if (work) work->edge_relaxations += reachable_edges;
        ++rounds;
        previous = dist;
        std::fill(next.begin(), next.end(), 0);
        bool changed = false;
        for (int u = 0; u < N; ++u) if (active[u]) {
            for (int e = row_offsets[u]; e < row_offsets[u + 1]; ++e) {
                const int v = col_indices[e], candidate = previous[u] + weights[e];
                if (candidate < dist[v]) {
                    if (dist[v] == INT_MAX)
                        reachable_edges += row_offsets[v + 1] - row_offsets[v];
                    dist[v] = candidate;
                    next[v] = 1;
                    changed = true;
                }
            }
        }
        if (!changed) break;
        active.swap(next);
    }
    return rounds;
}
