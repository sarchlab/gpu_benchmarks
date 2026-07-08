# tier2/heteromark_pagerank — PageRank (HeteroMark)

Benchmarks iterative PageRank (power method) on a **synthetic Erdős–Rényi graph** stored in CSR format. Emits per-iteration kernel times; throughput in edges per second is derived downstream.

Derived from the [HeteroMark benchmark suite](https://github.com/NUCAR-DEV/Hetero-Mark) PageRank workload.

## Algorithm

- **N** vertices (default: 65,536), average out-degree 16, random Erdős–Rényi graph generated at runtime with a deterministic PRNG (seed=42)
- Graph stored in CSR format (incoming edges for gather-based update)
- Iterative power method (20 iterations by default):
  - `PR[v] = (1-d)/N + d * sum(PR[u]/degree[u])` for all in-neighbors u of v
  - Damping factor d = 0.85
- Metric: `edges/sec = N_edges × PR_iterations / time_sec` (derived downstream from the emitted times)
- Each timed pass resets the rank vector to 1/N (untimed) before running all PageRank iterations, so every pass does identical work

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_vertices` | Number of vertices | 65536 |
| `BENCH_PARAM_pr_iterations` | PageRank iterations per pass | 20 |
| `BENCH_PARAM_damping_factor` | Damping factor d | 0.85 |
| `BENCH_PARAM_num_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed passes, each recorded | 7 |

## Files

| File | Description |
|------|-------------|
| `heteromark_pagerank.hip` | HIP source — PageRank gather kernel |
| `heteromark_pagerank.metal` | Metal compute shader — PageRank gather kernel |
| `heteromark_pagerank_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./heteromark_pagerank                                  # defaults: N=65536, 20 PR iters
./heteromark_pagerank --vertices 32768                 # smaller graph
./heteromark_pagerank --vertices 65536 --pr-iterations 30
```

## Output

**stdout** — JSON lines, one `kernel` record per timed pass (the accumulated
GPU event time of all PageRank iterations within the pass) followed by a
`summary` whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"pagerank_kernel","iter":0,"time_ms":45.678900,"host_time_ms":46.100000,"params":{"vertices":65536,"pr_iterations":20,"damping_factor":0.850000}}
...
{"type":"summary","total_time_ms":319.752300}
```

Derived metrics (e.g. edges/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Vertices: 65536  |  PR iterations: 20  |  Warmup: 3

Graph: 65536 vertices, 1048576 edges (avg in-degree: 16.0)
pass 0: pagerank_kernel gpu = 45.679 ms, host = 46.100 ms
...
Total GPU time: 319.752 ms over 7 passes
PASS
```

## Verification

Element-wise comparison of GPU PageRank values against CPU reference implementation with relative tolerance of 1e-4.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_pagerank.hip`; one kernel launch per PageRank iteration
- **Metal**: compiled with `clang++` from `heteromark_pagerank_metal.mm`; the Metal shader (`heteromark_pagerank.metal`) is loaded and compiled at runtime from the same directory as the binary
