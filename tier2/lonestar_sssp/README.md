# lonestar_sssp — Single-Source Shortest Path (Bellman-Ford)

## Provenance & fidelity

Derived from the [Lonestar GPU](https://iss.oden.utexas.edu/?p=projects/galois/lonestar) suite — the `sssp` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Topology-driven Bellman-Ford vs Lonestar's worklist SSSP; "inspired by" rather than a port. (Disclosed.)

Read this as "an SSSP benchmark in the spirit of Lonestar," not Lonestar's `sssp`.

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
| `--block_size` | 256   | Threads per block              |

Environment variables (`BENCH_PARAM_vertices`, `BENCH_PARAM_degree`,
`BENCH_PARAM_block_size`) override the flags. In addition:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed run; set `false` to time as-is |

## Build & Run

```bash
# Auto-detect platform
make
./lonestar_sssp

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
./lonestar_sssp --vertices 131072 --degree 8
```

## Output Format

**JSON lines to stdout:** the single timed run (Bellman-Ford until
convergence) emits one `kernel` record with the accumulated GPU event time
(`time_ms`) and host time (`host_time_ms`) of all Bellman-Ford iterations,
then a summary whose `total_time_ms` equals the recorded GPU event time:
```json
{"type":"kernel","name":"bellman_ford_kernel","iter":0,"time_ms":12.345,"host_time_ms":13.101,"params":{"vertices":65536,"degree":16,"block_size":256}}
{"type":"summary","total_time_ms":12.345}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
SSSP (Bellman-Ford)  |  Vertices: 65536  |  Edges: 1048576

gpu = 12.345 ms, host = 13.101 ms (8 BF iterations)
Total GPU time: 12.345 ms
PASS
```

## Metric (derived downstream)

**Medges/sec** = `N_edges × BF_iterations / time_seconds / 1e6`

Computed downstream from `time_ms` and params — not reported by the binary.

## Verification

Compares GPU shortest-path distances against a sequential CPU Bellman-Ford
reference for the first 1024 vertices. Reports PASS/FAIL; on FAIL the binary exits with a non-zero status.
