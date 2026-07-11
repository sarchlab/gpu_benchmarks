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
| `--degree`         | 16      | Average out-degree per vertex  |
| `--pr-iterations`  | 5       | PageRank iterations per run    |

Environment variables (`BENCH_PARAM_vertices`, `BENCH_PARAM_degree`,
`BENCH_PARAM_pr_iterations`) override the command-line flags. In addition:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed run; set `false` to time as-is |

**Generator note:** the graph is generated deterministically (fixed seed 42).
Each vertex draws about `degree` outgoing endpoints uniformly at random —
**self-loops and duplicate edges are permitted** and kept, so `degree` is the
real per-vertex out-degree used for rank division. Minimum out-degree is 1,
so there are no dangling nodes and the ranks form a probability distribution
(mass-consistent). The PageRank run executes exactly once per execution;
repetition across fresh processes is the harness's job
(`gpubench run --reps N`).

## Build & Run

```bash
# Auto-detect platform
make
./graph_pr

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
./graph_pr --vertices 65536 --degree 8 --pr-iterations 30
```

## Output Format

**JSON lines to stdout** — the single timed run emits one `kernel` record
carrying the accumulated GPU event time (`time_ms`) and host wall-clock time
(`host_time_ms`) of all PageRank iterations, then a `summary` whose
`total_time_ms` equals the recorded GPU event time:
```
{"type":"kernel","name":"pagerank_kernel","iter":0,"time_ms":45.123400,"host_time_ms":46.010000,"params":{"vertices":131072,"degree":16,"pr_iterations":5}}
{"type":"summary","total_time_ms":45.123400}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
PageRank (pull-based)  |  Vertices: 131072  |  Edges: 2097152  |  PR iterations: 5  |  Damping: 0.85

pagerank_kernel gpu = 45.123 ms, host = 46.010 ms
Total GPU time: 45.123 ms
PASS
```

## Metric

**GTEPS** (Giga Traversed Edges Per Second) = `N_edges × PR_iterations / time_seconds / 1e9`

Derived downstream from the emitted `time_ms` and params; the benchmark itself
reports raw times only.

## Verification

Compares GPU PageRank values against a sequential CPU reference implementation
for the first 1024 vertices using a scale-aware tolerance: a vertex fails if
`|gpu - cpu| > 1e-7 + 1e-2 * max(|cpu|, 1/N)` (ranks have magnitude ~1/N, so
an absolute tolerance would be vacuous at large N). Additionally checks mass
conservation: the total GPU rank must satisfy `|sum - 1.0| <= 1e-3` (the
generator guarantees min out-degree 1, so ranks form a probability
distribution). Reports PASS if both checks hold; a FAIL makes the benchmark
exit non-zero (after all JSON output is emitted).
