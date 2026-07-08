# tier2/graph_tc — Triangle Counting (Graph Analytics)

Counts the number of triangles in an undirected graph using GPU-accelerated sorted adjacency list intersection. Each GPU thread processes one edge and counts common neighbors via a merge-based intersection of sorted neighbor lists.

Derived from common graph analytics triangle counting workloads.

## Algorithm

- Generate a random undirected graph with **N** vertices and average degree **D** (default: N=16,384, D=32)
- Store graph in CSR (Compressed Sparse Row) format with sorted adjacency lists
- For each directed edge (u, v) with u < v, count common neighbors of u and v by intersecting their sorted neighbor lists
- Each triangle (a, b, c) is counted 3 times (once per edge), so divide total by 3
- triangles/sec = total\_triangles / time\_sec (derived downstream from the recorded times)

## Files

| File | Description |
|------|-------------|
| `graph_tc.hip` | HIP source — triangle counting kernel, graph generation, verification |
| `graph_tc.metal` | Metal compute shader — triangle counting kernel |
| `graph_tc_metal.mm` | ObjC++ Metal host code |
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
| `BENCH_PARAM_verify` | true | CPU verification (true, false, skip) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup launches (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed launches, each recorded |

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per timed iteration with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"triangle_count_kernel","iter":0,"time_ms":12.3456,"host_time_ms":12.4012,"params":{"vertices":16384,...}}
...
{"type":"summary","total_time_ms":86.4192}
```

Derived metrics (triangles/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Triangle Counting  |  Vertices: 16384  |  Avg Degree: 32  |  Iterations: 3 warmup + 7 timed

Generating random graph...
Graph: 16384 vertices, 524288 directed edges (CSR nnz)
Undirected edges (u<v): 262144
triangle_count_kernel iter 0: gpu = 12.346 ms, host = 12.401 ms
...
Total GPU time: 86.419 ms over 7 iterations
Triangles found:  12345
Running CPU verification...
PASS (CPU=12345, GPU=12345 triangles)
```

## Verification

- GPU triangle count is compared against a CPU reference implementation
- Both use the same sorted adjacency list intersection algorithm
- Reports PASS/FAIL with counts

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `graph_tc.hip`; one thread per edge
- **Metal**: compiled with `clang++` from `graph_tc_metal.mm`; the Metal shader (`graph_tc.metal`) is loaded and compiled at runtime from the same directory as the binary
