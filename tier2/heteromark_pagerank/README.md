# tier2/heteromark_pagerank — PageRank (HeteroMark)

Benchmarks iterative PageRank (power method) on a **synthetic Erdős–Rényi graph** stored in CSR format. Emits the run's kernel time; throughput in edges per second is derived downstream.

## Provenance & fidelity

Faithful reproduction of the [HeteroMark benchmark suite](https://github.com/NUCAR-DEV/Hetero-Mark) PageRank workload: genuine pull-based (gather) power-method PageRank.

## Algorithm

- **N** vertices (default: 65,536), average out-degree 16, random Erdős–Rényi graph generated at runtime with a deterministic PRNG (seed=42)
- Graph stored in CSR format (incoming edges for gather-based update)
- Iterative power method (5 iterations by default):
  - `PR[v] = (1-d)/N + d * sum(PR[u]/degree[u])` for all in-neighbors u of v
  - Damping factor d = 0.85 (default)
- Metric: `edges/sec = N_edges × PR_iterations / time_sec` (derived downstream from the emitted times)
- The rank vector is initialised to 1/N before the run (untimed); the run then executes all `pr_iterations` kernel launches

## Measurement

The PageRank power method (all `pr_iterations` launches) runs **exactly once**
per process execution (see the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_vertices` / `--vertices` | Number of vertices | 65536 |
| `BENCH_PARAM_pr_iterations` / `--pr-iterations` | PageRank iterations per run | 5 |
| `BENCH_PARAM_damping_factor` / `--damping_factor` | Damping factor d | 0.85 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `heteromark_pagerank.hip` | HIP source — PageRank gather kernel |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./heteromark_pagerank                                  # defaults: N=65536, 5 PR iters
./heteromark_pagerank --vertices 32768                 # smaller graph
./heteromark_pagerank --vertices 65536 --pr-iterations 10
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) with the GPU event
time accumulated over all PageRank iterations of the single run, followed by a
`summary` whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"pagerank_kernel","iter":0,"time_ms":11.419700,"host_time_ms":11.525000,"params":{"vertices":65536,"pr_iterations":5,"damping_factor":0.850000}}
{"type":"summary","total_time_ms":11.419700}
```

Derived metrics (e.g. edges/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Vertices: 65536  |  PR iterations: 5

Graph: 65536 vertices, 1048576 edges (avg in-degree: 16.0)
pagerank_kernel gpu = 11.420 ms, host = 11.525 ms
Total GPU time: 11.420 ms
PASS
```

## Verification

Element-wise comparison of GPU PageRank values against CPU reference implementation with relative tolerance of 1e-4. On FAIL the process exits non-zero.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_pagerank.hip`; one kernel launch per PageRank iteration
