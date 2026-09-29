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
{"type":"kernel","name":"bellman_ford_kernel","iter":0,"time_ms":12.345,"host_time_ms":13.101,"params":{"vertices":65536,"degree":16,"block_size":256,"verification":"full_cpu_distances_v1"}}
{"type":"summary","total_time_ms":12.345}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
SSSP (Bellman-Ford)  |  Vertices: 65536  |  Edges: 1048576

gpu = 12.345 ms, host = 13.101 ms (8 BF iterations)
Total GPU time: 12.345 ms
PASS: 0 mismatches in 65536 vertices
```

## Metric (derived downstream)

**Medges/sec** = `N_edges × BF_iterations / time_seconds / 1e6`

Computed downstream from `time_ms` and params — not reported by the binary.

## Verification

Every actual measured GPU distance is compared exactly against the complete
sequential CPU Bellman-Ford reference, including unreachable `INT_MAX` values.
There is no sampled prefix or verification-only GPU rerun. All mismatches are
counted, with at most ten vertex diagnostics; a mismatch exits nonzero. The
kernel record identifies this policy as `verification: "full_cpu_distances_v1"`.
The CPU reference, download and comparison are outside GPU timing but count
toward the runner's whole-process budget. The existing reference already
computed the entire distance vector before full comparison was enabled.

Regression commands from the repository root (use new work directories):

```bash
python3 tests/sssp_verification.py --host-only --compiler clang++ --work-dir /tmp/sssp-host
python3 tests/sssp_verification.py --platform cuda --sweep --work-dir /tmp/sssp-gpu
```

Host fixtures compare the production CPU reference with independent
Floyd-Warshall distances. GPU tests corrupt actual post-timing distances,
including the former cutoff and final vertex, check skipped-work rejection,
bounded diagnostics and both warmer modes. The sweep covers all 81 declared
vertex/degree configurations with a 60-second per-process limit. CI runs host
checks and compiles the GPU variants; their execution requires hardware.

Full final-distance coverage does not resolve the existing mixed ordinary and
atomic distance accesses, plain convergence-flag writes, or schedule-dependent
work. Those algorithm/concurrency and work-identity repairs remain separate
under [issue #71](https://github.com/sarchlab/gpu_benchmarks/issues/71) R-09/R-10.
This coverage change alone does not qualify SSSP for cross-GPU performance
comparisons.
