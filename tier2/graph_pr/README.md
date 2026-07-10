# graph_pr — Graph PageRank (Pull-Based)

## Algorithm

Computes PageRank on a random sparse directed graph using the iterative
pull-based method. The graph is stored in Compressed Sparse Column (CSC) format
for efficient pull access.

Each iteration launches one thread per vertex. Each thread computes its new
rank by summing contributions from all incoming neighbors:

```
PR[v] = (1 - d) / N + d * Σ (PR[u] / out_degree[u])  for all u → v
```

where `d = 0.85` is the damping factor.

## Parameters

| Parameter          | Default | Description                    |
|-------------------|---------|--------------------------------|
| `--vertices`       | 131072  | Number of graph vertices       |
| `--degree`         | 16      | Average edges per vertex       |
| `--pr-iterations`  | 20      | PageRank iterations per pass   |

Environment variables (`BENCH_PARAM_vertices`, `BENCH_PARAM_degree`,
`BENCH_PARAM_pr_iterations`) override the command-line flags. In addition:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_warmup` | 3 | Warmup PageRank passes (untimed) |
| `BENCH_PARAM_iteration`  | 7 | Timed PageRank passes, each recorded |

## Build & Run

```bash
# Auto-detect platform
make
./graph_pr

# Explicit platform
make PLATFORM=metal
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
./graph_pr --vertices 65536 --degree 8 --pr-iterations 30
```

## Output Format

**JSON lines to stdout** — one `kernel` record per timed pass carrying the
accumulated GPU event time (`time_ms`) and host wall-clock time
(`host_time_ms`) of all PageRank iterations in that pass, then a `summary`
whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"pagerank_kernel","iter":0,"time_ms":45.123400,"host_time_ms":46.010000,"params":{"vertices":131072,"degree":16,"pr_iterations":20}}
...
{"type":"summary","total_time_ms":315.863800}
```

**Human-readable to stderr:**
```
Device: Apple M1
PageRank (pull-based)  |  Vertices: 131072  |  Edges: 2097152  |  PR iterations: 20  |  Damping: 0.85  |  Passes: 3 warmup + 7 timed

pass 0: pagerank_kernel gpu = 45.123 ms, host = 46.010 ms
...
Total GPU time: 315.864 ms over 7 passes
PASS
```

## Metric

**GTEPS** (Giga Traversed Edges Per Second) = `N_edges × PR_iterations / time_seconds / 1e9`

Derived downstream from the emitted `time_ms` and params; the benchmark itself
reports raw times only.

## Verification

Compares GPU PageRank values against a sequential CPU reference implementation
for the first 1024 vertices. Reports PASS if all values match within 1e-4
absolute tolerance.
