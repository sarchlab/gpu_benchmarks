# lonestar_sssp — Single-Source Shortest Path (Bellman-Ford)

## Algorithm

Computes single-source shortest paths on a random sparse directed graph using
the Bellman-Ford algorithm on the GPU. The graph is stored in Compressed Sparse
Row (CSR) format with random integer edge weights (1–100).

Each iteration launches one thread per vertex. Each thread relaxes all outgoing
edges from its vertex using atomic min operations on the distance array. The
algorithm repeats until no distance is updated (convergence).

This is inspired by the Lonestar GPU benchmark suite's SSSP implementation.

## Parameters

| Parameter    | Default | Description                    |
|-------------|---------|--------------------------------|
| `--vertices` | 65536   | Number of graph vertices       |
| `--degree`   | 16      | Average edges per vertex       |
| `--iterations`| 5      | Legacy iteration-count param (echoed in records) |

Environment variables (`BENCH_PARAM_*`) override the flags. In addition:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_num_warmup` | 3 | Warmup SSSP passes (not recorded) |
| `BENCH_PARAM_num_iters`  | 7 | Timed SSSP passes, each recorded  |

## Build & Run

```bash
# Auto-detect platform
make
./lonestar_sssp

# Explicit platform
make PLATFORM=metal
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
./lonestar_sssp --vertices 131072 --degree 8
```

## Output Format

**JSON lines to stdout:** one `kernel` record per timed pass with the
accumulated GPU event time (`time_ms`) and host time (`host_time_ms`) of all
Bellman-Ford iterations in that pass, then a summary whose `total_time_ms`
is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"bellman_ford_kernel","iter":0,"time_ms":12.345,"host_time_ms":13.101,"params":{"vertices":65536,"degree":16,"iterations":5,"block_size":256}}
{"type":"summary","total_time_ms":86.42}
```

**Human-readable to stderr:**
```
Device: Apple M1
SSSP (Bellman-Ford)  |  Vertices: 65536  |  Edges: 1048576  |  Passes: 3 warmup + 7 timed

Bellman-Ford converged in 8 iterations
pass 0: gpu = 12.345 ms, host = 13.101 ms (8 BF iterations)
...
Total GPU time: 86.420 ms over 7 passes
PASS
```

## Metric (derived downstream)

**Medges/sec** = `N_edges × BF_iterations / time_seconds / 1e6`

Computed downstream from `time_ms` and params — not reported by the binary.

## Verification

Compares GPU shortest-path distances against a sequential CPU Bellman-Ford
reference for the first 1024 vertices. Reports PASS/FAIL.
