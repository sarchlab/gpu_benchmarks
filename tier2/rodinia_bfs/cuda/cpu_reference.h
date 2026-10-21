#pragma once
#include <vector>

struct BfsWork {
    unsigned long long levels = 0;
    unsigned long long frontier_edges = 0;
    unsigned long long discovery_atomics = 0;
    unsigned long long promotion_atomics = 0;
};

// Exact shortest-path distances from vertex 0 in an unweighted directed CSR.
// The caller supplies a valid nonempty graph with destinations in [0,n).
// Mark on enqueue: each vertex occupies at most one of the n queue slots,
// including graphs with cycles, self-loops, duplicate edges or unreachable nodes.
static std::vector<int> bfs_reference(const int* row, const int* col, int n, BfsWork* work = nullptr) {
    if (work) *work = BfsWork{};
    std::vector<int> distance(n, -1), queue(n);
    distance[0] = 0;
    queue[0] = 0;
    int head = 0, tail = 1;
    while (head < tail) {
        const int u = queue[head++];
        if (work) work->levels = static_cast<unsigned long long>(distance[u]) + 1;
        for (int at = row[u]; at < row[u+1]; ++at) {
            const int v = col[at];
            if (distance[v] == -1) {
                distance[v] = distance[u] + 1;
                queue[tail++] = v;
                if (work) ++work->promotion_atomics;
            }
            if (work) {
                ++work->frontier_edges;
                // These are precisely the destinations unvisited at u's level
                // entry, including duplicate edges and competing parents.
                if (distance[v] == distance[u] + 1) ++work->discovery_atomics;
            }
        }
    }
    return distance;
}
