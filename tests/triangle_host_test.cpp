#include "../tier2/graph_tc/graph_tc_host.h"
#include <cstdio>
#include <set>

static long long original_count(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    int N)
{
    long long total = 0;
    for (int u = 0; u < N; ++u) {
        for (int idx = row_ptr[u]; idx < row_ptr[u + 1]; ++idx) {
            int v = col_idx[idx];
            if (v <= u) continue; // only process edges u < v
            // Intersect neighbors of u and v
            int ui = row_ptr[u], ue = row_ptr[u + 1];
            int vi = row_ptr[v], ve = row_ptr[v + 1];
            while (ui < ue && vi < ve) {
                if (col_idx[ui] == col_idx[vi]) {
                    total++;
                    ui++;
                    vi++;
                } else if (col_idx[ui] < col_idx[vi]) {
                    ui++;
                } else {
                    vi++;
                }
            }
        }
    }
    // Each edge (u,v) with u<v counts ALL common neighbors, so a triangle
    // (a,b,c) is counted once per each of its three edges — 3 times total.
    // Return the raw count; the caller divides by 3 (same as the GPU path).
    return total;
}

// ---------------------------------------------------------------------------
// Graph generation: random undirected graph with sorted adjacency lists
// ---------------------------------------------------------------------------

static void original_generate(int N, int avg_degree,
                           std::vector<int>& row_ptr,
                           std::vector<int>& col_idx)
{
    long long target_edges = (long long)N * avg_degree / 2;
    std::set<std::pair<int,int>> edge_set;

    srand(42);
    while ((long long)edge_set.size() < target_edges) {
        int u = rand() % N;
        int v = rand() % N;
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        edge_set.insert({u, v});
    }

    // Build adjacency lists
    std::vector<std::vector<int>> adj(N);
    for (auto& e : edge_set) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }

    // Sort each adjacency list
    for (int i = 0; i < N; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
    }

    // Build CSR
    row_ptr.resize(N + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + (int)adj[i].size();
    }
    col_idx.resize(row_ptr[N]);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < (int)adj[i].size(); ++j) {
            col_idx[row_ptr[i] + j] = adj[i][j];
        }
    }
}


// Independent triple enumeration for the small-graph oracle.
static long long cubic(const std::vector<int>& row, const std::vector<int>& col, int n) {
    std::vector<bool> adjacent(n * n);
    for (int u = 0; u < n; ++u)
        for (int at = row[u]; at < row[u + 1]; ++at) adjacent[u * n + col[at]] = true;
    long long triangles = 0;
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            for (int w = v + 1; w < n; ++w)
                if (adjacent[u * n + v] && adjacent[u * n + w] && adjacent[v * n + w]) ++triangles;
    return 3 * triangles;
}

static bool check_table() {
    using namespace graph_tc_host;
    for (long long count : {0LL,1LL,2LL,3LL,4LL,7LL,8LL,9LL,67108864LL,67108865LL}) {
        const auto capacity = edge_capacity(count);
        if (!capacity || (capacity & (capacity - 1)) || capacity / 2 < static_cast<std::uint64_t>(count) ||
            (count ? capacity / 4 >= static_cast<std::uint64_t>(count) : capacity != 1)) return false;
    }
    for (long long bad : {-1LL, std::numeric_limits<long long>::max()}) {
        try { (void)edge_capacity(bad); return false; }
        catch (const std::length_error&) {}
    }
    for (int u : {0,1,65535,65536,1048575,std::numeric_limits<int>::max() - 1}) {
        for (int v : {1,65536,1048575,std::numeric_limits<int>::max()}) {
            if (u >= v) continue;
            const auto edge = pack_edge(u, v);
            if (edge == empty_edge || static_cast<int>(edge >> 32) != u ||
                static_cast<int>(static_cast<std::uint32_t>(edge)) != v) return false;
        }
    }
    // Four keys collide at the last slot, exercising wraparound and duplicates.
    std::vector<std::uint64_t> table(8, empty_edge), keys;
    for (int v = 1; v < 1000 && keys.size() < 4; ++v)
        if ((edge_hash(pack_edge(0, v)) & 7) == 7) keys.push_back(pack_edge(0, v));
    if (keys.size() != 4) return false;
    for (auto key : keys) if (!insert_edge(table, key)) return false;
    const auto saved = table;
    for (auto key : keys)
        if (insert_edge(table, key) || table != saved || std::count(table.begin(), table.end(), key) != 1) return false;
    return std::count(table.begin(), table.end(), empty_edge) == 4;
}

static bool check_generated(int n, int degree) {
    std::vector<int> old_row, old_col, row = {99,98}, col = {97,96,95};
    int old_next[16], next[16];
    original_generate(n, degree, old_row, old_col);
    for (int& value : old_next) value = std::rand();
    graph_tc_host::generate_graph(n, degree, row, col);
    for (int& value : next) value = std::rand();
    if (old_row != row || old_col != col || !std::equal(old_next, old_next + 16, next)) return false;
    const long long total = graph_tc_host::triangle_count_cpu(row, col, n);
    if (total != original_count(old_row, old_col, n)) return false;
    return n > 257 || total == cubic(row, col, n);
}

int main() {
    if (!check_table()) { std::fprintf(stderr, "Edge table check failed\n"); return 1; }
    int exhaustive = 0, generated = 0;
    for (int n = 1; n <= 6; ++n) {
        const int edges = n * (n - 1) / 2;
        for (unsigned mask = 0; mask < (1u << edges); ++mask) {
            std::vector<std::vector<int>> adjacent(n);
            int bit = 0;
            for (int u = 0; u < n; ++u) {
                for (int v = u + 1; v < n; ++v, ++bit) {
                    if (mask & (1u << bit)) { adjacent[u].push_back(v); adjacent[v].push_back(u); }
                }
            }
            std::vector<int> row(n + 1), col;
            for (int u = 0; u < n; ++u) {
                std::sort(adjacent[u].begin(), adjacent[u].end());
                col.insert(col.end(), adjacent[u].begin(), adjacent[u].end());
                row[u + 1] = static_cast<int>(col.size());
            }
            const auto actual = graph_tc_host::triangle_count_cpu(row, col, n);
            if (actual != original_count(row, col, n) || actual != cubic(row, col, n)) {
                std::fprintf(stderr, "Triangle oracle failed n=%d mask=%u\n", n, mask);
                return 1;
            }
            ++exhaustive;
        }
    }
    for (int n : {2,3,4,5,7,16,31,64,128,257}) {
        const std::set<int> degrees = {1,2,8,32,128,n - 1};
        for (int degree : degrees) {
            if (degree <= 0 || degree >= n) continue;
            if (!check_generated(n, degree)) {
                std::fprintf(stderr, "Dataset/RNG/reference mismatch n=%d degree=%d\n", n, degree);
                return 1;
            }
            ++generated;
        }
    }
    // Sparse large-ID cases keep the host CI check small while crossing 16 bits.
    for (auto pair : {std::pair<int,int>{65537,1}, {65539,8}}) {
        if (!check_generated(pair.first, pair.second)) {
            std::fprintf(stderr, "Large vertex-ID dataset mismatch\n");
            return 1;
        }
        ++generated;
    }
    if (exhaustive != 33867 || generated != 38) return 1;
    std::printf("%d exhaustive graphs, %d datasets/RNG continuations and edge-table checks pass\n", exhaustive, generated);
    return 0;
}
