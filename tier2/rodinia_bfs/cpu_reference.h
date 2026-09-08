#pragma once
#include <vector>

// Exact shortest-path distances from vertex 0 in an unweighted directed CSR.
// The caller supplies a valid nonempty graph with destinations in [0,n).
// Mark on enqueue: each vertex occupies at most one of the n queue slots,
// including graphs with cycles, self-loops, duplicate edges or unreachable nodes.
static std::vector<int> bfs_reference(const int* row, const int* col, int n) {
    std::vector<int> distance(n, -1), queue(n);
    distance[0] = 0;
    queue[0] = 0;
    int head = 0, tail = 1;
    while (head < tail) {
        const int u = queue[head++];
        for (int at = row[u]; at < row[u+1]; ++at) {
            const int v = col[at];
            if (distance[v] == -1) {
                distance[v] = distance[u] + 1;
                queue[tail++] = v;
            }
        }
    }
    return distance;
}
