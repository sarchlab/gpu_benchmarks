# lonestar_sssp — synchronous topology-driven Bellman–Ford

## Provenance and input

This portable reimplementation follows the topology-driven relaxation in
[Lonestar GPU / GaloisGPU at `5b6ff19`](https://github.com/IntelligentSoftwareSystems/GaloisGPU/blob/5b6ff19b364fda39a80bda9920e1edb59e17086f/apps/sssp/main.cu).
That historical version visits outgoing edges and uses atomic minimum distance
updates. Later Lonestar versions use different, worklist-based algorithms;
this benchmark does not implement those variants.

Two differences are intentional and owner-approved:

- Upstream updates distances in place within a launch. Here each round reads
  an immutable snapshot, so thread scheduling cannot change logical propagation
  rounds. The required snapshot copy is a timed GPU kernel.
- Upstream reads a graph file. Here the existing deterministic synthetic CSR
  graph is retained: xorshift32 seed 42, per-vertex degree
  `max(1, floor(D/2) + rng % (D+1))`, destinations `rng % N`, integer weights
  `1 + rng % 100`, and source vertex 0. It is not an upstream dataset.

Input generation, all 81 declared vertex/degree configurations, and defaults
are unchanged. Results from the former asynchronous implementation must be
recollected; they do not describe this workload schedule.

## Algorithm and timing

Each round first copies the current distances to a separate immutable array.
One relaxation thread per vertex then attempts an `atomicMin` for every outgoing
edge whose source distance was finite at round entry. The result array retains
previous minima. A thread signals convergence with `atomicExch` if any candidate
beats the **snapshot** destination distance. Both reads and this predicate are
independent of competing atomic update order. Positive weights ensure convergence
within N rounds, including the terminating no-change round.

`sssp_snapshot_kernel` and `bellman_ford_kernel` each have an accumulated GPU-event
and host-lap record. Their summed GPU time is the summary. Allocation, graph
upload, initial distances, convergence-flag transfers and full CPU verification
are outside GPU kernel timing. The entry warmer runs once, after initialization
and before the first snapshot. Data reuse between subsequent kernels is part of
the algorithm. No extra GPU computation is run for verification.

## Parameters and execution

| Parameter | Default | Meaning |
|---|---:|---|
| `--vertices` | 65536 | Number of vertices |
| `--degree` | 16 | Degree generator parameter |
| `--block_size` | 256 | Threads per block |

`BENCH_PARAM_vertices`, `BENCH_PARAM_degree` and `BENCH_PARAM_block_size` override
flags. `BENCH_PARAM_vector_add_warmup` defaults to `true`; `false` disables the
entry warmer without promising a particular cache/clock state.

Resolved vertices and block size must be positive; degree must be nonnegative.
An environment degree of zero retains the generator's one-edge-per-vertex case.
The existing positive-only CLI fallback and environment precedence are unchanged.
Before allocation or GPU access, a sufficient integer-safety bound requires
`N * max(1, floor(D/2) + D) <= INT_MAX` for every possible generated CSR prefix,
and `100 * N < INT_MAX` for every distance candidate, including the terminating
round. This limits N to 21,474,836 and preserves all 81 declared configurations.
Grid rounding uses wide arithmetic. With this N bound, even padded thread
indices fit signed int for any positive int block size. These checks do not
guarantee memory capacity, runtime budget or hardware support for arbitrary
manual block sizes.

```bash
make PLATFORM=cuda  # or PLATFORM=rocm
./lonestar_sssp --vertices 131072 --degree 16
```

## Work descriptors and verification

Both kernel records carry the same parameters:

- `input: "xorshift32_seed42_v1"` and `algorithm: "topology_snapshot_v1"` identify
  the graph generator and synchronous schedule.
- `verification: "full_cpu_distances_rounds_v2"` identifies complete exact integer
  comparison and expected-round verification.
- `graph_edges` is the actual generated edge count; `iterations` is the actual
  measured round count, including the terminating no-change round.
- `edge_relaxations` counts outgoing edges from all vertices reachable at each
  round's entry, summed across rounds. Each corresponds to one GPU `atomicMin`
  attempt. Unreachable sources contribute zero. This is deterministic logical
  work derived by the CPU reference, not a hardware counter or the number of
  successful atomics. It is valid for passing runs, whose final distances and
  round count are verified. Do not multiply `graph_edges * iterations`: early
  rounds skip unreachable sources. Do not sum the duplicated work descriptor
  across the two kernel records.

Downstream attempted Medges/s can use `edge_relaxations / total_gpu_seconds / 1e6`,
including the required snapshot cost. The binary reports raw timings and counts.

The independent CPU reference propagates only sources whose values changed in
the previous round, with a full distance snapshot and sources scanned in CSR
order. This reduces CPU reference work while producing the same synchronous
trajectory. It separately tracks all currently reachable outgoing edges to
calculate the GPU's logical work. Every actual measured final distance,
including unreachable `INT_MAX`, is compared exactly; a round-count mismatch
also fails. All distance mismatches are counted, at most ten are printed, and
failures exit nonzero while retaining timing records. There is no sampled check
or relaxed floating-point tolerance. CPU verification and downloads count toward
the whole-process timeout.

## Regression checks

From the repository root, with new work directories:

```bash
python3 tests/sssp_verification.py --host-only --compiler clang++ --work-dir /tmp/sssp-host
python3 tests/sssp_verification.py --platform cuda --sweep --work-dir /tmp/sssp-gpu
python3 tests/sssp_rounds.py --binary /tmp/sssp-gpu/faults/tier2/lonestar_sssp/hip/lonestar_sssp --work-dir /tmp/sssp-rounds
```

Host tests cover 32,768 exhaustive graph/source/weight cases against independent
Floyd–Warshall distances and dense snapshot round/work counts, plus weighted,
self/parallel/unreachable and forward/reverse chain fixtures. The device sweep
checks all 81 configurations in both warmer modes with a 60-second process limit,
automatically rejects differing work descriptors between modes,
corrupts actual final distances and rejects skipped launches. The round tests
compare every intermediate GPU vector with an independent Python model and
instrument actual atomic attempts in a separate test copy. They vary block size,
repeat runs and reject a wrong round count even when final distances are correct.
CI runs host tests and compiles device variants; execution requires hardware.
CUDA sanitizer checks are supplementary: racecheck alone does not prove absence
of global-memory races; separation of immutable reads and atomic writes provides
the concurrency argument.

This change addresses the SSSP algorithm/work contract. Arbitrary manual-input
and capacity admission, whole-suite qualification and AMD hardware validation
remain separate. It does not establish publication-quality performance on a
shared H100.
