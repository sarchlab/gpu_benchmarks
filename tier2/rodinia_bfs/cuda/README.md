# tier2/rodinia_bfs/cuda — Breadth-First Search (Rodinia BFS), native CUDA

Iterative BFS on a randomly-generated CSR (Compressed Sparse Row) graph.
Two GPU kernels are launched per BFS level, iterated until the traversal
completes. Reports GPU event time for irregular graph traversal.

This directory is a self-contained native CUDA build. Apart from the shared
timing/warmup headers in `common_cuda/`, it depends on nothing outside itself.

## Provenance & fidelity

The algorithm reference is Rodinia 3.1 at commit
`366b283456506ef2fe2c2b7dc5e83e941ffc5524`:
[expansion kernel](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/bfs/kernel.cu),
[frontier promotion](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/bfs/kernel2.cu), and
[driver](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/bfs/bfs.cu).
This independent implementation keeps source-0, unit-edge breadth-first
traversal, runs expansion followed by frontier promotion at each level, and
terminates when no new vertices are found. Like the pinned expansion kernel,
it tests the visited mask from level entry.

The local driver generates a directed random CSR graph instead of reading the
pinned input format. It stores offsets rather than Node records and uses int
masks rather than bool masks. The separate updating mask is represented by a
newly assigned distance together with the visited mask. Frontier clearing occurs
in promotion. These choices preserve the recurrence and stages but change
layout and traffic. Competing distance and completion-flag stores use atomic
exchange to repair the unsynchronized stores in the original algorithm.
Kernel instructions, memory traffic and performance are therefore not expected
to match the original implementation.

## Architecture requirements

The kernels use only plain global loads/stores and 32-bit `atomicExch`. There
are no warp intrinsics, shared memory, tensor cores or other
microarchitecture-specific features, so the code runs on any sm_70+ GPU.
Its intended targets are **A100 (sm_80)** and **H100 (sm_90)**.

## Algorithm

The graph is stored in Compressed Sparse Row (CSR) format:
- `row_offsets[i]` and `row_offsets[i+1]` delimit the edges of node `i`
- `col_indices[e]` gives the destination node of edge `e`
- `cost[v]` holds the BFS level of node `v` (−1 if not yet visited)

Each BFS level executes two kernel passes:

**Pass 1 — bfs_kernel**: For each node in the current frontier, atomically store `cost[node] + 1` for every neighbor unvisited at level entry. Eligibility does not change during expansion, so competing parents and duplicate edges each make their own atomic attempt.

**Pass 2 — bfs_update**: Promote nodes that were just discovered (`cost ≥ 0` but not yet `visited`) into the next frontier and set `visited[v] = 1`.

Both passes are iterated until no new nodes are added to the frontier. The
benchmark runs **one complete traversal, exactly once**; repetition is the
harness's job.

## Parameters

| Parameter | Env var | Default | Notes |
|-----------|---------|---------|-------|
| `num_nodes` | `BENCH_PARAM_num_nodes` (or `--num_nodes N`) | 65536 | number of graph nodes (scaling axis) |
| `edges_per_node` | `BENCH_PARAM_edges_per_node` (or `--edges_per_node E`) | 20 | out-degree of each node |
| `block_size` | `BENCH_PARAM_block_size` (or `--block_size B`) | 256 | GPU thread-block size |
| `vector_add_warmup` | `BENCH_PARAM_vector_add_warmup` | `true` | cold-cache calibration before the timed traversal |

`params.json` declares the scaling sweep (`num_nodes`, 1024 … 8388608) and the
non-scaling values (`edges_per_node` ∈ {5, 20, 80}, `block_size` = 256,
`vector_add_warmup` ∈ {true, false}).

Resolved `num_nodes` and `block_size` must be positive, and
`edges_per_node` must be nonnegative. Zero-edge graphs are still supported
through the environment override. The CLI parser, however, falls back to its
default when a flag is nonpositive. The edge count and every padded launch
thread's vertex index must fit in a signed `int`; out-of-range inputs fail
before any allocation or device access. Row-offset byte sizes and launch
rounding use wide arithmetic. These checks guarantee representable indexing.
They do not guarantee that enough host/device memory is available, or that the
hardware supports an arbitrary block size.

## Files

| File | Description |
|------|-------------|
| `rodinia_bfs.cu`   | CUDA source: BFS kernels, graph generator and driver |
| `cpu_reference.h`  | Independent complete CPU distance reference |
| `params.json`      | Scaling / non-scaling parameter declaration |
| `Makefile`         | nvcc build |
| `../../../common_cuda/bench_timing.h`, `bench_warmup.h` | Shared timing harness and cold-cache warmer (native CUDA) |

## Build

Requires the CUDA Toolkit (`nvcc`). `-arch=native` needs CUDA ≥ 11.5 and a GPU
visible at build time.

```bash
cd tier2/rodinia_bfs/cuda
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
make clean
```

## Run

```bash
./rodinia_bfs                                        # defaults: 65536 nodes, 20 edges/node
./rodinia_bfs --num_nodes 131072
BENCH_PARAM_num_nodes=65536 BENCH_PARAM_edges_per_node=5 ./rodinia_bfs
BENCH_PARAM_vector_add_warmup=false ./rodinia_bfs    # skip the cold-cache warmer
```

A quick sweep over the declared scaling axis (default non-scaling values):

```bash
for n in 1024 16384 262144 1048576 8388608; do
  BENCH_PARAM_num_nodes=$n ./rodinia_bfs
done
```

The `gpubench` harness does not yet discover `cuda/` subdirectories, so
`gpubench run` still builds and runs the original source one level up.
[TODO for cuda]

## Output

**stdout**: JSON lines. There is one `kernel` record per kernel, carrying that
kernel's GPU event time accumulated over all BFS levels of the traversal. A
final `summary` record follows, whose `total_time_ms` is the sum of the
recorded times:

```
{"type":"kernel","name":"bfs_kernel","iter":0,"time_ms":...,"host_time_ms":...,"params":{"num_nodes":65536,"edges_per_node":20,"block_size":256,"verification":"full_cpu_v1","discovery_policy":"level_visited_atomic_v1","bfs_levels":6,"frontier_edges":1310720,"discovery_atomics":233381,"promotion_atomics":65535}}
{"type":"kernel","name":"bfs_update","iter":0,"time_ms":...,"host_time_ms":...,"params":{...}}
{"type":"summary","total_time_ms":...}
```

The work counts above come from the H100 host input generator; other input
generators may produce different counts. Rates can be derived downstream from
`time_ms` and the logical counts. These counts do not measure physical memory
traffic or bandwidth.

**stderr**: device info, node/edge counts, per-kernel timing, and full-distance
verification. A mismatch prints `FAIL` and exits nonzero. Every distance is
compared, but only the first ten mismatches are printed.

## Correctness

After timing, every measured distance is downloaded and compared exactly with
an independent serial CPU breadth-first traversal (`cpu_reference.h`) of the
same directed graph. The reference starts at vertex 0, uses unit edge lengths,
and keeps `-1` for unreachable vertices. Cycles, self-loops and duplicate edges
do not change the shortest-distance contract. Both kernel records report
`full_cpu_v1`.

The CPU reference costs O(V+E) and uses two V-element integer vectors for
distances and its queue (64 MiB total at 8,388,608 vertices). The result
download reuses the existing host cost buffer and transfers 32 MiB at that
size. Both happen outside the recorded timing intervals and add no GPU kernel
launches.

The discovery policy is `level_visited_atomic_v1`. During expansion the visited
mask is read-only. Every frontier vertex is already visited, so its cost is only
read. Only unvisited vertices receive cost writes, and all of them go through
atomic exchange. Promotion runs after expansion completes and owns each
vertex's frontier and visited entries. Its shared completion flag also uses
atomic exchange. Kernel ordering separates all other conflicting accesses
between levels.

Both records include the actual `bfs_levels`, checked against the CPU
reference, and logical work counts derived from that independent traversal:
- `frontier_edges` counts the outgoing edges of reachable vertices.
- `discovery_atomics` counts edges from level d to level d+1, including
  duplicates.
- `promotion_atomics` counts newly reached vertices other than the source.

These counts do not depend on GPU scheduling for a given graph. They describe
algorithm operations, not physical cache transactions or hardware-counter
measurements.

## Qualification and tests

These items exist for the original source but have not yet been ported to or
run against this CUDA build:

- CPU-reference oracle test (`tests/bfs_reference_test.cpp`) — it checks the
  algorithm-only `cpu_reference.h`, which is byte-identical here. [TODO for cuda]
- Measured-result test with post-timing fault injection
  (`tests/bfs_measured_result.py`). [TODO for cuda]
- Per-level work-count instrumentation test (`tests/bfs_work_counts.py`). [TODO for cuda]
- Input-range tests (`tests/bfs_input_ranges.py`). [TODO for cuda]
- Host-adapter concurrency/sanitizer test (`tests/bfs_concurrent_kernels.py`). [TODO for cuda]
- CI compile job for `cuda/`. [TODO for cuda]
- `gpubench` harness discovery of `cuda/`. [TODO for cuda]
- H100 qualification of all 81 declared configurations, GPU sanitizer and
  timing-order checks, and A100 qualification. [TODO for cuda]
