# tier2/rodinia_bfs — Breadth-First Search (Rodinia BFS)

Iterative BFS on a randomly-generated CSR (Compressed Sparse Row) graph.
Two GPU kernels are launched per BFS level, iterated until the traversal
completes. Measures effective memory bandwidth of irregular graph traversal.

## Provenance & fidelity

The algorithm reference is Rodinia 3.1 at commit
`366b283456506ef2fe2c2b7dc5e83e941ffc5524`:
[expansion kernel](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/bfs/kernel.cu),
[frontier promotion](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/bfs/kernel2.cu), and
[driver](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/bfs/bfs.cu).
This independent implementation retains source-0, unit-edge breadth-first
traversal, expansion followed by frontier promotion at each level, and
termination when no new vertices are found. The pinned expansion tests the
visited mask from level entry; this implementation does the same.

The local driver generates a directed random CSR graph instead of reading the
pinned input format. It stores offsets rather than Node records and uses int
masks rather than bool masks. The separate updating mask is represented by a
newly assigned distance together with the visited mask. Frontier clearing occurs
in promotion. These choices preserve the recurrence and stages but change
layout and traffic. Competing distance and completion-flag stores use atomic
exchange to repair the unsynchronized stores in the original algorithm.
Kernel instructions, memory traffic and performance are therefore not expected
to match the original implementation. Input portability remains a separate
qualification requirement.

## Algorithm

The graph is stored in Compressed Sparse Row (CSR) format:
- `row_offsets[i]` and `row_offsets[i+1]` delimit the edges of node `i`
- `col_indices[e]` gives the destination node of edge `e`
- `cost[v]` holds the BFS level of node `v` (−1 if not yet visited)

Each BFS level executes two kernel passes:

**Pass 1 — bfs_kernel**: For each node in the current frontier, atomically store `cost[node] + 1` for every neighbor unvisited at level entry. Eligibility does not change during expansion, so competing parents and duplicate edges each perform their own atomic attempt.

**Pass 2 — bfs_update**: Promote nodes that were just discovered (`cost ≥ 0` but not yet `visited`) into the next frontier and set `visited[v] = 1`.

Both passes are iterated until no new nodes are added to the frontier. The
benchmark runs **one complete traversal, exactly once**; repetition is the
harness's job (`gpubench run --reps N`).

## Parameters

| Parameter | Env var | Default | Notes |
|-----------|---------|---------|-------|
| `num_nodes` | `BENCH_PARAM_num_nodes` (or `--num_nodes N`) | 65536 | number of graph nodes (scaling axis) |
| `edges_per_node` | `BENCH_PARAM_edges_per_node` (or `--edges_per_node E`) | 20 | out-degree of each node |
| `block_size` | `BENCH_PARAM_block_size` (or `--block_size B`) | 256 | GPU thread-block size |
| `vector_add_warmup` | `BENCH_PARAM_vector_add_warmup` | `true` | cold-cache calibration before the timed traversal |

## Files

| File | Description |
|------|-------------|
| `rodinia_bfs.hip`        | HIP source — BFS kernels + self-contained utilities |
| `Makefile`               | Auto-detecting build (ROCm→rocm, else→cuda) |
| `cpu_reference.h` | Independent complete CPU distance reference |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./rodinia_bfs                                        # defaults: 65536 nodes, 20 edges/node
./rodinia_bfs --num_nodes 131072
BENCH_PARAM_num_nodes=65536 BENCH_PARAM_edges_per_node=5 ./rodinia_bfs
```

## Output

**stdout** — JSON lines: one `kernel` record per kernel, carrying that
kernel's GPU event time accumulated over all BFS levels of the traversal,
then a `summary` whose `total_time_ms` is the sum of the recorded times:

```
{"type":"kernel","name":"bfs_kernel","iter":0,"time_ms":...,"host_time_ms":...,"params":{"num_nodes":65536,"edges_per_node":20,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"bfs_update","iter":0,"time_ms":...,"host_time_ms":...,"params":{...}}
{"type":"summary","total_time_ms":...}
```

Bandwidth / edges-per-second are derived downstream from `time_ms` and the
params — the benchmark never computes them.

**stderr** — device info, node/edge counts, per-kernel timing, and full-distance
verification. A mismatch prints `FAIL` and exits nonzero; all distances are
compared even though individual diagnostics are limited to the first ten errors.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_bfs.hip`
- Input data is generated synthetically — no external input files required

## Correctness and qualification

Every measured distance is downloaded after timing and compared exactly with
an independent serial CPU breadth-first traversal of the same directed graph.
The reference starts at vertex 0, uses unit edge lengths, and preserves `-1`
for unreachable vertices. Cycles, self-loops and duplicate edges do not change
the shortest-distance contract. Both kernel rows record `full_cpu_v1`.

The CPU reference costs O(V+E) and uses two V-element integer vectors for
distances and its queue (64 MiB total at 8,388,608 vertices). The added result
download reuses the existing host cost buffer and transfers 32 MiB at that
size. These operations are outside the recorded timing intervals and add no
GPU kernel launches. Include this host work and transfer in target process
budgets; AMD runtime qualification remains separate.

The discovery policy is `level_visited_atomic_v1`. During expansion the visited
mask is read-only. Every frontier vertex is already visited, so its cost is only
read; only unvisited vertices receive cost writes, all through atomic exchange.
Promotion runs after expansion completes and owns each vertex's frontier and
visited entries. Its shared completion flag uses atomic exchange. Kernel
ordering separates all other conflicting accesses between levels.

Both records include the actual `bfs_levels`, checked against the CPU reference,
and logical work derived from that independent traversal: `frontier_edges`
counts outgoing edges of reachable vertices, `discovery_atomics` counts edges
from level d to level d+1 (including duplicates), and `promotion_atomics` counts
newly reached vertices other than the source. These counts are invariant under
GPU scheduling for the same graph. They describe algorithm operations, not
physical cache transactions or a hardware-counter measurement. Changing the
graph or its input generator can change the counts.

`tests/bfs_reference_test.cpp` checks the CPU reference against an independent
all-pairs shortest-path and logical-work oracle on all 66,066 directed graphs through four
vertices, including self-loops, plus 25 structured cases with duplicate edges
and disconnected components. CI runs it under host sanitizers.

`tests/bfs_measured_result.py` checks actual GPU results on random and structured
graphs, including unreachable vertices, and injects post-timing device faults
at the source, middle and tail. Structured graphs replace inputs only in the
test copy; they are not production collection configurations. Run `python3
tests/bfs_measured_result.py --platform cuda --work-dir /tmp/bfs-check` from the
repository root with a new, nonexistent output directory (or use `rocm`). CI
compiles this GPU test; execution requires a GPU. A timeout fails the test and
retains completed logs and a partial report.
