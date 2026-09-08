# tier2/graph_tc — Triangle Counting (Graph Analytics)

Counts the number of triangles in an undirected graph using GPU-accelerated sorted adjacency list intersection. Each GPU thread processes one edge and counts common neighbors via a merge-based intersection of sorted neighbor lists.

## Provenance & fidelity

Not named after a specific upstream suite — derived from common graph-analytics triangle-counting workloads. Faithful to the algorithm it claims: sorted-adjacency-list (merge) intersection triangle counting.

## Algorithm

- Generate a random undirected graph with **N** vertices and average degree **D** (default: N=16,384, D=32)
- Store graph in CSR (Compressed Sparse Row) format with sorted adjacency lists
- For each edge (u, v) with u < v, count **all** common neighbors of u and v by intersecting their sorted neighbor lists
- Each triangle (a, b, c) is counted 3 times (once per edge), so divide total by 3
- triangles/sec = total\_triangles / time\_sec (derived downstream from the recorded times)

## Files

| File | Description |
|------|-------------|
| `graph_tc.hip` | HIP source — triangle counting kernel, graph generation, verification |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./graph_tc                    # defaults: N=16384, D=32
./graph_tc N=8192 D=16        # smaller graph
./graph_tc N=32768 D=64       # larger graph
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_vertices` | 16384 | Number of vertices |
| `BENCH_PARAM_avg_degree` | 32 | Average degree per vertex |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed launch; set `false` to time as-is |

`BENCH_PARAM_verify=false` (also `0`/`skip`) skips CPU verification — an
undeclared debug knob, not a sweep parameter in `params.json`. The CPU check
is expensive at large graph sizes. Every kernel record now identifies the
actual policy: `raw_triangle_total_cpu_v1` for the full CPU total comparison,
or `disabled` for this existing manual bypass. A bypassed run is unverified;
large-graph process budgets and broader override policy remain separate.

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record for the single timed launch with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` equals that launch's GPU event time:
```
{"type":"kernel","name":"triangle_count_kernel","iter":0,"time_ms":12.3456,"host_time_ms":12.4012,"params":{"vertices":16384,...}}
{"type":"summary","total_time_ms":12.3456}
```

Derived metrics (triangles/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Triangle Counting  |  Vertices: 16384  |  Avg Degree: 32

Generating random graph...
Graph: 16384 vertices, 524288 directed edges (CSR nnz)
Undirected edges (u<v): 262144
triangle_count_kernel: gpu = 12.346 ms, host = 12.401 ms
Total GPU time: 12.346 ms
Triangles found:  12345
Running CPU verification...
PASS (CPU=12345, GPU=12345 triangles)
```

## Verification

- The exact raw GPU contribution sum is compared against the full CPU raw sum,
  before division by three is used for triangle-count diagnostics. Comparing
  only the integer quotients previously accepted raw errors of one or two.
- Both use the same sorted adjacency list intersection algorithm
- Reports PASS/FAIL with counts; on FAIL the binary exits with a non-zero status

`python3 tests/triangle_total_verification.py --platform cuda --work-dir /tmp/tc-check`
from the repository root injects signed errors into actual device edge counts
after timing, covering first/middle/last positions, all declared degrees, a
two-vertex zero-triangle case, valid results and manual bypass reporting. Choose a new, nonexistent `--work-dir`
for each invocation. Use `--platform rocm` for AMD; CI only compiles the test.
A test timeout stops the suite with failure; completed-case logs are retained.

The GPU kernel, graph/input generator, launch, transfers, timing and full CPU
reference algorithm are unchanged. This corrects result acceptance without
reducing the existing CPU verification cost (R-17). Cross-device execution and
input qualification remain necessary; no AMD execution is claimed here.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `graph_tc.hip`; one thread per edge
