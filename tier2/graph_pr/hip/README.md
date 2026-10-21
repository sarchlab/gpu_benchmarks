# graph_pr — Graph PageRank (Pull-Based)

## Provenance & fidelity

Not named after a specific upstream suite — a generic graph-analytics PageRank. Faithful to the algorithm it claims: a genuinely pull-based (CSC gather) power-method PageRank.

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
**self-loops and duplicate edges are permitted** and kept. Each vertex's stored
out-degree counts every outgoing edge and is used for rank division; the
`degree` parameter controls the random draw, rather than fixing every vertex
to that exact degree. Minimum out-degree is 1,
so there are no dangling nodes and the ranks form a probability distribution
(mass-consistent). The PageRank run executes exactly once per execution;
repetition across fresh processes is the harness's job
(`gpubench run --reps N`).

Resolved vertices and iteration count must be positive, and degree must be
nonnegative. Invalid requests fail before graph allocation or GPU access and
emit no timing records, so zero-step runs cannot count as successful measurements.
An environment degree of zero still generates one outgoing edge per vertex.
The existing positive-only CLI fallback and environment precedence are unchanged.

A sufficient integer-safety bound requires
`N * max(1, floor(D/2) + D) <= INT_MAX`, covering generated degrees and all CSC
offsets. Row sizing and grid rounding use wide arithmetic; the fixed 256-thread
blocks keep even padded vertex indices within signed int. All 81 declared
configurations are unchanged. These checks do not guarantee memory capacity or
the process budget for arbitrary manual sizes/iteration counts. The numerical
tolerance limitation below remains separate.

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
{"type":"kernel","name":"pagerank_kernel","iter":0,"time_ms":45.123400,"host_time_ms":46.010000,"params":{"vertices":131072,"degree":16,"pr_iterations":5,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":45.123400}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
PageRank (pull-based)  |  Vertices: 131072  |  Edges: 2097152  |  PR iterations: 5  |  Damping: 0.85

pagerank_kernel gpu = 45.123 ms, host = 46.010 ms
Total GPU time: 45.123 ms
Max error (131072 vertices): 1.000000e-11
PageRank verification: PASS (131072 vertices, 0 mismatches)
```

## Metric

**GTEPS** (Giga Traversed Edges Per Second) = `num_edges × pr_iterations / time_seconds / 1e9`

Use the actual edge count printed on stderr, the `pr_iterations` parameter and
the emitted kernel time. The benchmark itself reports raw times only.

## Verification

Every actual final GPU rank is compared with the complete sequential CPU
trajectory for the same requested iteration count. There is no second GPU solve.
Both values must be finite, and the existing local per-vertex criterion remains
`|gpu - cpu| <= 1e-7 + 1e-2 * max(|cpu|, 1/N)`. The kernel record identifies this
check with `verification: "full_cpu_v1"`.

The independent mass check requires `|sum(gpu) - 1.0| <= 1e-3`; the generator
has no dangling vertices. Conserving mass does not establish individual ranks:
opposite errors in two vertices can cancel. Both checks must pass for a zero
exit code. The diagnostic counts every vertex and mismatch, printing at most ten
individual examples. Download, CPU computation and comparison are all untimed.

The fixed `1e-7` term can dominate ranks near `1/N` for large graphs and admit
large relative per-vertex errors. This coverage repair retains that numerical
policy; tightening it is a separate follow-up. Complete coverage alone does not
remove this limitation.

From the repository root, using new output directories:

```bash
python3 tests/graph_pr_verification.py --host-only --work-dir /tmp/pr-host
python3 tests/graph_pr_verification.py --platform cuda --work-dir /tmp/pr-gpu --sweep
```

Host cases use exact analytical trajectories, duplicate edges, a one-vertex
graph with a self-loop, and total mass. GPU cases check startup and
partial blocks, odd/even final-buffer selection, balanced errors after the former
1024-vertex cutoff, nonfinite ranks, both warmer settings and bounded diagnostics.
`--sweep` additionally runs all 81 declared configurations with the unmodified
production binary and warmer enabled, requiring complete verification within
60 seconds per process.

On the NVIDIA H100 PCIe with CUDA on 2026-09-28, all 116 production/fault
cases passed, including all 81 declared configurations. The largest case
(8,388,608 vertices, degree 64, five iterations) completed in 46.638 seconds
including graph generation and complete verification. Four Compute Sanitizer
tools, eight nonfinite-reference/output checks and six native CLI/SQLite
success/failure checks passed. The compiled PageRank kernel's complete SASS
section is identical to the previous implementation. These shared-GPU checks
establish correctness coverage and observed runtime budget, not publication
performance or AMD qualification; the tolerance limitation above remains.
