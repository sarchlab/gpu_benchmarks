# cache_bandwidth — Repeated-window read bandwidth (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

Measures the logical read bandwidth of blocks repeatedly streaming vectorized
`float4` loads across a window. Sweep the window size to study the memory
hierarchy; the timing alone does not identify the cache level serving a load.

## Workload and topology

Each block reads every vector in its window `iters` times and writes each
thread's accumulator to a sink. The inner repetition count targets approximately
8 GiB of logical reads per launch, with a minimum of 3 and maximum of 4,000,000.
The minimum can make large launches exceed the target. The byte count includes
all passes, including the first fill, and excludes sink stores and the warmer.
It describes requested loads, not measured DRAM traffic.

- `shared` (default): every block reads the same window. The requested grid is
  8 times the reported CU/SM count, with 256 threads per block.
- `private`: every launched block reads a distinct window. The requested grid
  is the reported CU/SM count, with 1024 threads per block. If the allocation
  cap cannot fit all requested windows, the grid is reduced:

  ```text
  cap = min(free_device_memory / 2, 4 GiB)
  num_blocks = windows = min(requested_num_blocks, floor(cap / working_set_bytes))
  allocation_bytes = windows * working_set_bytes
  ```

  The kernel's `blockIdx.x % windows` therefore equals `blockIdx.x`. If even
  one window cannot fit, the process refuses the configuration. For example,
  114 requested blocks with a 1 GiB window and a 4 GiB cap launch **4 blocks**,
  each with its own window. Earlier binaries launched 114 blocks sharing those
  four windows; those results describe different work and must be recollected.

Distinct windows do not assign blocks to particular CUs/SMs or guarantee private
cache residency. `blocks_per_cu` is a requested grid multiplier, not an observed
placement or occupancy. Dividing aggregate bandwidth by `num_blocks` gives an
average per launched block, not a measured per-CU/SM bandwidth.

## Controls and records

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./cache_bandwidth                     # 1 MiB shared window
BENCH_PARAM_mode=private ./cache_bandwidth 262144
BENCH_PARAM_mode=private BENCH_PARAM_threads_per_block=256 ./cache_bandwidth
```

| Control | Default | Meaning |
|---|---|---|
| `working_set_bytes` | 1048576 | Per-window bytes, at least 16 and a multiple of 16. Environment overrides positional argument 1. |
| `mode` | shared | Exact `private` selects distinct windows; other direct-binary strings retain the shared fallback. The manifest accepts only shared/private. |
| `threads_per_block` | 256 shared / 1024 private | Direct-binary, unswept environment override. Integers 64 through 1024 are honored; other parsed values retain the mode default. Device launch limits still apply. |
| `blocks_per_cu` | 8 shared / 1 private | Direct-binary, unswept environment override. Positive parsed integers replace the mode default; others retain it. |
| `vector_add_warmup` | true | Suite-wide clock/cache warmer; set false to disable. |

Use the `BENCH_PARAM_` prefix for each environment control. The two geometry
overrides are intentionally not Cartesian sweep axes. The harness can inherit
these environment variables; unset them for default collection. The manifest
retains 21 window sizes (1 KiB through 1 GiB) times two modes: 42 configurations.

Every timed kernel record retains `working_set_bytes`, `mode`, `blocks_per_cu`,
`num_blocks`, and `iters`, and additionally records:

| Descriptor | Meaning |
|---|---|
| `requested_num_blocks` | Resolved `blocks_per_cu` times the runtime CU/SM count, before the memory cap |
| `threads_per_block` | Actual block size after override/default resolution |
| `windows` | Number of distinct allocated windows (1 in shared mode) |
| `allocation_bytes` | Read-buffer allocation, excluding the sink and warmer |
| `bytes_per_iter` | `num_blocks * working_set_bytes`, using the actual grid |
| `bytes_read` | `bytes_per_iter * iters`, total logical read bytes in the timed launch |
| `launch_policy` | `shared_window_v1` or `distinct_windows_v1` |

Compute decimal GB/s as `bytes_read / (time_ms * 1e6)`. Compare raw launch times
across devices only with equivalent effective work. For bandwidth comparisons,
report the full geometry and topology alongside the byte-normalized rate;
a reduced grid may leave much of a device idle. Reproduce a recorded workload
only when the target can launch the same geometry and fit the same windows.
The source guards loop/sink indices and byte arithmetic before allocation.
The existing shared half-free-memory limit and private no-window refusal remain.

## Measurement and validation

Each process launches and times `bw_read_kernel` exactly once with GPU events,
then emits its raw kernel record and the matching summary. Unless disabled,
the suite warmer runs immediately before timing. The first read pass is included
in the measurement; this benchmark has no untimed resident-cache primer.
Use fresh processes (`gpubench run --reps N`) for statistical repetitions.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. `tests/cache_bandwidth_geometry.py` exercises the production control resolution,
cap boundaries, indexing limits and complete JSON descriptors with simulated
device limits under fatal host sanitizers. `tests/cache_bandwidth_gpu.py` checks
actual GPU records and the unchanged kernel's distinct/shared window addressing
using nonuniform test inputs. GPU acceptance does not establish physical cache
hit rates.