# shared_mem_bandwidth — Tier 1 microbenchmark (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

This benchmark times dependent shared-memory read/update/write traffic with two
address layouts. Each selected kernel runs once per process. Results are raw GPU
and host times; derive logical bandwidth downstream after qualifying the compiled
work and the target's bank behavior.

| Pattern | Addresses within each logical group of 32 lanes |
|---|---|
| `no_conflict` | Consecutive four-byte words |
| `conflict` | Distinct four-byte words, 32 words (128 bytes) apart |

Both patterns visit every element of the 48 KiB shared array exactly once per
inner iteration. The conflict layout transposes each 32×32 tile: for logical
index `i`, its physical index is
`(i / 1024) * 1024 + (i % 32) * 32 + (i % 1024) / 32`.
This permutation gives every element one owning thread. Sharing a bank therefore
does not mean sharing an address or introducing a data race.

## Work and timing

Both kernels use the same dependent recurrence, starting with `acc = 0` and
shared elements equal to one:

```cpp
acc = 0.5f * (acc + smem[index]);
smem[index] = acc;
```

Values remain in `[0,1]` throughout the sweep. Volatile shared accesses preserve
the requested loads and stores; every thread writes one final global checksum.
The timed kernel includes shared initialization, two barriers, indexing and
arithmetic, and final stores. It is a dependent traffic experiment; the result
does not establish peak shared-memory throughput.

The inner-loop logical traffic is
`2 * 12288 * 4 * inner_iters * num_blocks` bytes. Divide by recorded seconds and
`1e9` for logical GB/s. This count excludes the initialization stores and final
global stores even though their time is included. Overhead matters at small
iteration counts. Physical transactions and serialization require target evidence.

The vector-add warmer runs outside timing when enabled, with the intent of
raising clocks and disturbing cache state. Qualify that state on the target.
Allocation and test verification remain outside benchmark timing. Repeat runs
use fresh processes (`gpubench run --reps N`).

Earlier kernels used an unbounded sum and only retained one output per block;
the earlier conflict mapping also aliased addresses. These results describe a
changed experiment. Recollect measurements and keep revision/build provenance
when comparing with historical results.

## Bank interpretation

For NVIDIA's 32-bank, four-byte-word layout, consecutive words use different
banks; the conflict pattern places 32 distinct words in the same bank within
one warp. These are the assumptions behind the historical pattern names.
[NVIDIA's shared-memory guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
describes that mapping.

Every kernel record includes the actual device `warp_size` and the fixed layout
and footprint descriptors. A 64-lane wave contains two logical 32-lane groups;
the code does not infer a hardware bank count from wave size.

## Build and validate

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
make test-pattern        # CPU-only ownership and bank-model checks
make test  # also run the production-kernel GPU output test
make clean
```

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. The device needs at least 48 KiB of shared memory per block. The CPU test checks
complete unique ownership for every 32-thread multiple through 1024 and the
32/64-bank address models. The GPU test checks both patterns, every declared
block size, an uneven 160-thread distribution, the minimum/maximum declared
iteration counts, and one/two blocks.
It verifies every finite output and both buffer guards. Tests run separately
and add no verification launch to the measurement executable. CI runs the CPU
test and compiles the GPU test; actual GPU execution is a separate check.

## Parameters and output

| Environment parameter | Default | Meaning |
|---|---|---|
| `BENCH_PARAM_inner_iters` | 1000 | Positive inner-loop iteration count |
| `BENCH_PARAM_block_size` | 256 | Positive multiple of 32 within the device limit |
| `BENCH_PARAM_num_blocks` | 1 | Positive number of blocks |
| `BENCH_PARAM_access_pattern` | `no_conflict` | `no_conflict`, `conflict`, or standalone `both` |
| `BENCH_PARAM_vector_add_warmup` | `true` | Enable the untimed warmer |

The declared sweep keeps one block. A device-wide claim needs a separately
qualified grid and measurement; multiplying by SM/CU count is an extrapolation.
See [measurement interpretation](../../../docs/measurement_interpretation.md).

Run `./shared_mem_bandwidth` with these environment overrides. Progress goes to
stderr; stdout contains one kernel record per selected pattern and one summary.
For `both`, each record retains the requested `access_pattern=both`, with its
own kernel name and lane stride. Illustrative single-pattern output:

```json
{"type":"kernel","name":"smem_bw_conflict","iter":0,"time_ms":0.42,"host_time_ms":0.5,"params":{"inner_iters":64,"block_size":256,"num_blocks":1,"access_pattern":"conflict","shared_memory_bytes":49152,"word_bytes":4,"logical_lane_group_size":32,"lane_stride_bytes":128,"warp_size":32}}
{"type":"summary","total_time_ms":0.42}
```

`time_ms` covers the one kernel launch with GPU events. `host_time_ms` covers the
host bracket around event recording, submission, error checking and completion.
The summary is the sum of all emitted kernel GPU times. The extra footprint and
layout descriptors are measured/derived output metadata, not environment controls.
