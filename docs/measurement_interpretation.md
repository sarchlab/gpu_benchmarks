# Interpreting Tier 1 measurements

Tier 1 records raw timing and workload parameters. Derived rates describe the
selected configuration and the qualified executable. Keep the full configuration,
measurement controls, and source run together when comparing devices; use a
minimum successful whole-run GPU time across fresh-process repetitions, as in
[the output protocol](output_protocol.md).

## Single-block arithmetic and device-wide throughput

The FP16/FP32/FP64/INT32 manifests currently fix `num_blocks=1`. Report these as
**single-block arithmetic measurements**, with `threads_per_block` and the work
count. Per-SM/CU interpretation additionally depends on the target's scheduling
unit and how much of it the selected block occupies. One block does not establish
saturation of that unit or the whole device.

Multiplying a single-block rate by the device's SM/CU count is an extrapolation
model. It is neither a device-wide measurement nor a demonstrated upper bound:
occupancy, issue limits, clock behavior, and power limits can change with the
number of active blocks. A device-wide claim needs a separately declared grid
configuration/profile and a measured result at that grid size.

For a qualified binary that preserves the intended work, the nominal arithmetic
counts are:

| Benchmark | Count for one launch |
|---|---|
| FP32/FP64 FMA | `num_blocks * threads_per_block * fmas_per_thread * 2` FLOPs |
| Packed FP16 FMA | `num_blocks * threads_per_block * fmas_per_thread * 4` FLOPs |
| INT32 multiply-add | `num_blocks * threads_per_block * mads_per_thread * 2` integer operations |

Use effective reported work counts: positive FMA/MAD requests are rounded down
to a multiple of four with a minimum of four. All declared arithmetic values
already meet that constraint. These are source-level operation counts. Check the compiled instructions and
observable results before using them as measured throughput. Packed FP16 FMA
counts describe the `__hfma2` workload, not a matrix/tensor operation. Work
preservation and target instruction qualification remain R-02/R-19 in
[the readiness tracker](https://github.com/sarchlab/gpu_benchmarks/issues/71).

## Primed pointer-chase entry

`cache_latency` deliberately primes its chain after the optional clock/cache
warmer and before its one timed launch. This is the resident-cache exception to
default cold entry. Keep `cache_entry`, `primer_launches`, `warmup_accesses` and
`num_accesses` with the result. A capped primer, separate launch placement and
compiled load policy prevent inferring a guaranteed cache level or hit rate from
array size alone. A timed launch can still incur an initial L1 fill. See the
[benchmark methodology](../tier1/cache_latency/hip/README.md) for target limits and
trace replay: preserve the primer's state, exclude its work from measured latency.

## Cold and primed code-fetch entry

`code_fetch_latency` times the same kernel from two entries. Both first load
the selected kernel without running it, so runtime code loading is outside the
timed bracket. `prime_launches=0` is default cold entry: the timed launch is
the kernel's first dispatch in the process. `prime_launches=1` adds one untimed
launch of that kernel after the optional warmer, the primed exception to cold
entry. The primer stores to its own buffer, so the timed launch does not reuse
a primer's output. Pair the two results of one `body_ops` (minimum over
repetitions of each) and subtract; fit the difference against code bytes
measured from the compiled kernels, not against `body_ops`. The difference is
first-dispatch minus primed-dispatch time. Reading it or its slope as
instruction-fetch cost requires target qualification: the primer may have run
on a different SM/CU, address translation and other first-use state also
differ, and whether the attribute query fully pre-loads a kernel is
runtime-specific.
Keep `code_entry`, `prime_launches`, `reps` and `total_ops` with the result.
Neither entry certifies which cache level holds the code, and code larger than
the instruction cache makes the primed run refetch. See the
[benchmark methodology](../tier1/code_fetch_latency/hip/README.md) for the
compiled-code check, limits and trace replay.

## Empty dispatch and timing resolution

`empty_kernel.time_ms` is the GPU event interval around one empty dispatch.
It includes the dispatch/grid completion behavior and any delay between the
recorded events; it is not an isolated host launch API cost. The shared helper's
`host_time_ms` spans event recording, launch submission, error checking, stop-event
recording, and synchronization. Label it **host bracket/round-trip time**, rather
than pure launch-call latency.

Short intervals are sensitive to event granularity and scheduling noise. NVIDIA
documents approximately 0.5 microsecond resolution for
[`cudaEventElapsedTime`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).
Record the timer/runtime used and qualify its practical resolution on each target;
the number of printed decimal places does not establish measurement precision.
Preserve zero or near-resolution samples as timing observations, and flag their
rates as unresolved instead of dividing by zero or claiming infinite throughput.
Increasing repetitions characterizes variability; it does not improve timer
resolution. A larger workload changes the experiment and needs its own identity.

## Transfer payload, host allocation, and overlap

The `memory_bandwidth` manifest collects D2D only, with pinned host setup
buffers. Its direct-binary H2D/D2H/`all` modes are outside that sweep. Use
`pcie_bandwidth` for the declared host-transfer sweep with both allocation
types. For D2D, distinguish payload rate `B / (time_ms * 1e6)` from logical
read-plus-write rate `2 * B / (time_ms * 1e6)`, where
`B = buffer_size_mb * 1024 * 1024`. Logical bytes do not establish physical
HBM/GDDR traffic or peak bandwidth.

For `pcie_bandwidth`, let `B = buffer_size_mb * 1024 * 1024` bytes per direction.
The unidirectional payload rate is `B / (time_ms * 1e6)` in SI GB/s. The
bidirectional aggregate payload rate is `2 * B / (time_ms * 1e6)`; both transfers
use the same event bracket. This is a payload rate, not a count of every bus
transaction or host staging copy. Do not sum the three mode times and call that
a per-direction transfer measurement.

The manifest sweeps both pinned and pageable host memory. Keep these results
separate. The bidirectional path submits H2D and D2H on two streams; actual overlap
depends on memory allocation, runtime behavior, stream semantics, and the device.
NVIDIA documents that pageable async transfers may synchronize or require staging
in its [API synchronization rules](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).
AMD documents synchronous behavior for unpinned host memory in the
[`hipMemcpyAsync` reference](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html).
Consequently, the pageable bidirectional case measures the submitted pair with
its staging/serialization effects; concurrency must be established by profiling.

The current event bracket relies on legacy default-stream ordering with the two
blocking copy streams. Qualify completion ordering and overlap on the selected
runtime/build before treating its interval as an aggregate transfer measurement.
Pinned buffers permit asynchronous submission, but do not alone prove overlap or
a particular full-duplex bandwidth. Report the actual link/topology and measured
payload rate instead of substituting a nominal PCIe generation rate.

## Mixbench's nominal intensity model

The local [kernel](../tier1/mixbench/hip/mixbench.hip) loads one `float`, runs a dependent
`fmaf` loop, and stores one `float` for each active element. The host computes
`num_fmas = max(1, int(4 * flops_per_byte))`. For the declared ratios:

| Requested ratio | FMA calls/element | Nominal FLOPs/element | Logical bytes/element | Nominal FLOP/byte |
|---|---:|---:|---:|---:|
| 0.5 | 2 | 4 | 8 | 0.5 |
| 2 | 8 | 16 | 8 | 2 |
| 4 | 16 | 32 | 8 | 4 |
| 8 | 32 | 64 | 8 | 8 |
| 32 | 128 | 256 | 8 | 32 |

Here `elements = buffer_size_mb * 1024 * 1024 / 4`; total nominal FLOPs are
`elements * num_fmas * 2`, and logical read-plus-write bytes are `elements * 8`.
The arithmetic intensity is `num_fmas / 4`, including the truncation/clamp for
custom ratios. The current output rounds the requested ratio to one decimal,
so use the declared ratios for reproducible analysis until effective override
reporting is addressed under R-16.

Logical bytes are a source-level model. Cache hits, writeback, transaction
formation, and compiler transformations determine traffic at each physical memory
level; target instruction/traffic qualification belongs to R-19. A dependent chain
and a large buffer do not guarantee peak compute throughput or a memory-bound
regime. Classify the observed bottleneck from qualified measurements.

## Sampling guidance

Keep short sweeps when they cover meaningful cases. The five EP sizes, seven MG
sizes, and eight Parboil histogram sizes remain valid with advisory density
warnings. Add points when they resolve a behavior or capacity transition on the
target hardware, and keep runs within memory and process-time budgets. Hardware
and simulator profiles are tracked separately under R-15/R-22.

The September 7, 2026 manifests enumerate 1,584 Tier 1 and 5,341 Tier 2
configurations: 6,925 total, or 48,475 fresh processes at seven repetitions.
Measurement controls are fixed rather than multiplied into this count. Recalculate
after any profile or manifest-value change:

```bash
gpubench run --tier 1 --reps 7 --dry-run
gpubench run --tier 2 --reps 7 --dry-run
```
