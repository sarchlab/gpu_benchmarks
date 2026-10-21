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

Threads whose labels change set the shared global convergence flag with a
device-scope `atomicExch`. The host resets the flag before launching each
iteration and reads it after that kernel completes. This preserves the Jacobi
updates and stopping rule while avoiding concurrent non-atomic writes. The
atomic operation is part of the timed propagation kernel; recollect kernel
performance after this synchronization repair.

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

Resolved vertex count, iteration cap and block size must be positive; average
degree must be nonnegative. Invalid requests fail before graph allocation or
GPU access and emit no timing records. Positive caps still allow a verified
partial trajectory without convergence. An environment degree of zero retains
the minimum-one sampled endpoint per vertex, including existing self-edge
removal and symmetric-edge deduplication. Environment `atoi` behavior is unchanged.

A sufficient integer-safety bound requires
`2 * N * max(1, floor(D/2) + D) <= INT_MAX` before allocating adjacency lists.
It covers both directions of each sampled edge before deduplication, generated
degrees and every CSR prefix. The guard uses unsigned 64-bit arithmetic; grid
rounding is also widened. This bound keeps row sizes and padded vertex indices
within signed int and preserves all 69 declared configurations. It does not
guarantee memory capacity, manual block support or arbitrary runtime budgets.

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
all label-propagation iterations until convergence or the cap, then a summary whose
`total_time_ms` equals the recorded GPU event time:
```json
{"type":"kernel","name":"cc_propagate_kernel","iter":0,"time_ms":15.234,"host_time_ms":16.001,"params":{"num_vertices":100000,"avg_degree":16,"max_iterations":100,"block_size":256,"verification":"full_cpu_v1","iterations_performed":12,"converged":true}}
{"type":"summary","total_time_ms":15.234}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
Connected Components (label propagation)  |  Vertices: 100000  |  Edges: 3186470  |  ...

gpu = 15.234 ms, host = 16.001 ms (12 CC iterations) converged
Total GPU time: 15.234 ms
CC verification: PASS (100000 vertices, 0 mismatches)
```

## Metrics (derived downstream)

Derived downstream from `time_ms` and params — not reported by the binary:

- **MTEPS** (Mega Traversed Edges Per Second) = `num_edges × iterations_performed / time_seconds / 1e6`

## Verification

After timing, every actual GPU label is compared exactly with the complete CPU
Jacobi reference. Both paths stop at convergence or `max_iterations`, so this
also verifies the defined partial trajectory when the cap is reached. A capped
trajectory is not a claim of converged connected components. There is no second
GPU solve. The kernel record includes `verification: "full_cpu_v1"`.

`iterations_performed` records the number of timed propagation launches, while
`converged` is true only if the last executed iteration observed no label
changes. A no-change iteration at exactly `max_iterations` is converged. If the
last allowed iteration still changed labels, `converged` is false, even if those
new labels happen to be a fixed point. The binary does not run another iteration
to decide. Both fields are preserved in the CLI's `kernel_results.params_json`;
they describe the measured solve and do not by themselves prove verification
passed. A capped trajectory can still pass the complete CPU comparison.

The diagnostic reports all vertices and the complete mismatch count, with at
most ten individual examples. Any mismatch returns a nonzero exit code. The
reference, comparison and final download remain outside the timed kernels.

From the repository root, using new output directories:

```bash
python3 tests/graph_cc_verification.py --host-only --work-dir /tmp/cc-host
python3 tests/graph_cc_verification.py --platform cuda --work-dir /tmp/cc-gpu --sweep
```

The host test checks the production CPU routine against exact trajectories on a
path, a disconnected pair and an isolated vertex. GPU cases cover odd/even caps,
early convergence, convergence detection exactly at the cap, result descriptors,
wrong final-buffer selection, corruption past the old 1024
label cutoff and bounded diagnostics. `--sweep` additionally executes all 69
declared configurations with the unmodified production binary and warmer enabled,
requiring complete verification within 60 seconds per process. These checks
provide correctness and observed process-budget evidence, not publication
performance or AMD qualification.

H100 PCIe / CUDA 12.9 validation on 2026-09-28 passed all 69 configurations and
97 production/fault cases. The largest case verified all 1,000,000 labels; the
slowest whole process took under 11 seconds. Clang/GCC fatal host sanitizers and
eight native CLI/SQLite outcomes passed, including capped, converged-at-cap and
failed-verification records under both warmer settings. The shared GPU's
timings are correctness-budget evidence, not publication performance results.
The descriptor addition preserves the atomic-flag repair's compiled GPU kernel
instructions and all 95 common cases' iteration counts.
