# graph_cc — Graph Connected Components (Label Propagation)

## Algorithm

Computes connected components on a random undirected sparse graph using
iterative label propagation. The graph is stored in Compressed Sparse Row
(CSR) format.

Each vertex starts with its own ID as its component label. In each iteration,
every vertex updates its label to the minimum of its own label and all
neighbor labels:

```
for each vertex v in parallel:
    new_label = labels[v]
    for each neighbor u of v:
        new_label = min(new_label, labels[u])
    labels[v] = new_label
```

Iterations continue until no labels change (convergence) or the maximum
number of iterations is reached.

## Parameters

| Parameter         | Default  | Description                              |
|------------------|----------|------------------------------------------|
| `num_vertices`    | 100000   | Number of graph vertices                 |
| `avg_degree`      | 16       | Average edges per vertex                 |
| `max_iterations`  | 100      | Max label propagation iterations per run |
| `block_size`      | 256      | Threads per block                        |
| `iterations`      | 5        | Legacy iteration-count param (echoed in records) |
| `num_warmup`      | 3        | Warmup CC passes (not recorded)          |
| `num_iters`       | 7        | Timed CC passes, each recorded           |

Parameters are read from `BENCH_PARAM_*` environment variables.

## Build & Run

```bash
# Auto-detect platform
make
./graph_cc

# Explicit platform
make PLATFORM=metal
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
BENCH_PARAM_num_vertices=500000 BENCH_PARAM_avg_degree=8 ./graph_cc
```

## Output Format

**JSON-lines to stdout:** one `kernel` record per timed pass with the
accumulated GPU event time (`time_ms`) and host time (`host_time_ms`) of all
label-propagation iterations in that pass, then a summary whose
`total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"cc_propagate_kernel","iter":0,"time_ms":15.234,"host_time_ms":16.001,"params":{"num_vertices":100000,...}}
{"type":"summary","total_time_ms":76.17}
```

**Human-readable to stderr:**
```
Device: Apple M1
Connected Components (label propagation)  |  Vertices: 100000  |  Edges: 1600000  |  ...

pass 0: gpu = 15.234 ms, host = 16.001 ms (12 CC iterations)
...
Total GPU time: 76.170 ms over 7 passes
PASS
```

## Metrics (derived downstream)

Derived downstream from `time_ms` and params — not reported by the binary:

- **MTEPS** (Mega Traversed Edges Per Second) = `num_edges × cc_iterations / time_seconds / 1e6`

## Verification

Compares GPU labels against a sequential CPU reference implementation for
the first 1024 vertices. Reports PASS if all labels match exactly.
