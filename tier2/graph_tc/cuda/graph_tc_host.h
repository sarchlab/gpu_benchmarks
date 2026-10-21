#ifndef GRAPH_TC_HOST_H
#define GRAPH_TC_HOST_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace graph_tc_host {
// Valid vertex IDs are nonnegative ints, so this cannot encode an edge.
constexpr std::uint64_t empty_edge = std::numeric_limits<std::uint64_t>::max();

inline std::uint64_t pack_edge(int u, int v) {
    return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint32_t>(v);
}

inline std::uint64_t edge_hash(std::uint64_t key) {
    key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
    return key ^ (key >> 31);
}

// A power-of-two table with at most half its slots occupied. Check before
// doubling so impossible capacities fail instead of wrapping or probing forever.
inline std::size_t edge_capacity(long long expected) {
    if (expected < 0) throw std::length_error("negative edge count");
    const std::size_t maximum = std::vector<std::uint64_t>().max_size();
    std::size_t capacity = 1;
    while (capacity / 2 < static_cast<std::uint64_t>(expected)) {
        if (capacity > maximum / 2) throw std::length_error("edge table capacity");
        capacity *= 2;
    }
    return capacity;
}

// The table is nonempty, has power-of-two size and an unused slot; key is
// not empty_edge. generate_graph maintains these conditions at <= half load.
inline bool insert_edge(std::vector<std::uint64_t>& edges, std::uint64_t key) {
    const std::size_t mask = edges.size() - 1;
    std::size_t at = edge_hash(key) & mask;
    while (edges[at] != empty_edge && edges[at] != key) at = (at + 1) & mask;
    if (edges[at] == key) return false;
    edges[at] = key;
    return true;
}

// Preserve the original srand/rand calls, duplicate rejection and sorted CSR.
// As in the original generator, the requested edge count must be feasible for
// a simple undirected graph. General manual-input admission is separate.
inline void generate_graph(int n, int degree, std::vector<int>& row,
                           std::vector<int>& col) {
    if (n < 0) throw std::length_error("negative vertex count");
    const long long target = static_cast<long long>(n) * degree / 2;
    std::vector<std::uint64_t> edges(edge_capacity(target), empty_edge);
    long long used = 0;
    std::srand(42);
    while (used < target) {
        int u = std::rand() % n;
        int v = std::rand() % n;
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        if (insert_edge(edges, pack_edge(u, v))) ++used;
    }

    row.assign(static_cast<std::size_t>(n) + 1, 0);
    for (auto edge : edges) {
        if (edge == empty_edge) continue;
        ++row[static_cast<int>(edge >> 32) + 1];
        ++row[static_cast<int>(static_cast<std::uint32_t>(edge)) + 1];
    }
    for (int i = 0; i < n; ++i) row[i + 1] += row[i];
    col.resize(row[n]);
    std::vector<int> next = row;
    for (auto edge : edges) {
        if (edge == empty_edge) continue;
        const int u = static_cast<int>(edge >> 32);
        const int v = static_cast<int>(static_cast<std::uint32_t>(edge));
        col[next[u]++] = v;
        col[next[v]++] = u;
    }
    for (int i = 0; i < n; ++i)
        std::sort(col.begin() + row[i], col.begin() + row[i + 1]);
}

// For sorted, duplicate-free undirected CSR, count each u<v<w triangle once.
// Multiply by three to retain the full raw contribution total used by the GPU
// and the existing exact-total verification policy. No outputs are sampled.
inline long long triangle_count_cpu(const std::vector<int>& row,
                                    const std::vector<int>& col, int n) {
    std::vector<int> forward(n);
    for (int u = 0; u < n; ++u)
        forward[u] = static_cast<int>(std::upper_bound(col.begin() + row[u],
                                    col.begin() + row[u + 1], u) - col.begin());
    long long triangles = 0;
    for (int u = 0; u < n; ++u) {
        for (int at = forward[u]; at < row[u + 1]; ++at) {
            const int v = col[at];
            int ui = at + 1, vi = forward[v];
            const int ue = row[u + 1], ve = row[v + 1];
            while (ui < ue && vi < ve) {
                if (col[ui] == col[vi]) {
                    ++triangles;
                    ++ui;
                    ++vi;
                } else if (col[ui] < col[vi]) {
                    ++ui;
                } else {
                    ++vi;
                }
            }
        }
    }
    return 3 * triangles;
}
}  // namespace graph_tc_host

#endif
