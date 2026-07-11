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
| `avg_degree`      | 16       | Edge-draw parameter (see note below)     |
| `max_iterations`  | 100      | Max label propagation iterations per run |
| `block_size`      | 256      | Threads per block                        |
| `vector_add_warmup` | true   | Cold-cache calibration: boost the clock and evict the LLC before the timed run; set `false` to time as-is |

Parameters are read from `BENCH_PARAM_*` environment variables.

**Generator note:** each vertex draws roughly `avg_degree` random neighbors,
and every drawn edge is inserted **symmetrically** (into both endpoints'
adjacency lists, then deduplicated). The *actual* mean degree of the generated
undirected graph is therefore about **2× `avg_degree`** — unlike `graph_pr` /
`graph_tc`, where the parameter is the real per-vertex out-degree. The
propagation runs exactly once per execution; repetition across fresh processes
is the harness's job (`gpubench run --reps N`).

## Build & Run

```bash
# Auto-detect platform
make
./graph_cc

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
BENCH_PARAM_num_vertices=500000 BENCH_PARAM_avg_degree=8 ./graph_cc
```

## Output Format

**JSON-lines to stdout:** the single timed run emits one `kernel` record with
the accumulated GPU event time (`time_ms`) and host time (`host_time_ms`) of
all label-propagation iterations until convergence, then a summary whose
`total_time_ms` equals the recorded GPU event time:
```json
{"type":"kernel","name":"cc_propagate_kernel","iter":0,"time_ms":15.234,"host_time_ms":16.001,"params":{"num_vertices":100000,...}}
{"type":"summary","total_time_ms":15.234}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
Connected Components (label propagation)  |  Vertices: 100000  |  Edges: 3186470  |  ...

gpu = 15.234 ms, host = 16.001 ms (12 CC iterations)
Total GPU time: 15.234 ms
PASS
```

## Metrics (derived downstream)

Derived downstream from `time_ms` and params — not reported by the binary:

- **MTEPS** (Mega Traversed Edges Per Second) = `num_edges × cc_iterations / time_seconds / 1e6`

## Verification

Compares GPU labels against a sequential CPU reference implementation for
the first 1024 vertices. Reports PASS if all labels match exactly; on FAIL the binary exits with a non-zero status.
