# Local global-memory microbenchmarks (`shoc_devicememory`)

Three ordinary device-memory kernels measure read, write and in-place read/write
access. Each enabled kernel runs once, and every produced result is checked
against an exact FP32 CPU reference outside its timed interval.

## Scope and identity

The owner selected these existing three local kernels on 2026-09-28 instead of
restoring the full SHOC DeviceMemory suite. The directory, executable, suite tag
and benchmark ID remain `shoc_devicememory` for compatibility; they are historical
identifiers, not a claim of SHOC algorithm or performance equivalence. This is
an independent local global-memory microbenchmark.

SHOC DeviceMemory is a collection of access experiments, including shared memory
and textures. Those experiments are outside this selected scope. Ordinary
`hipMalloc` allocations are used here; on NVIDIA, the compatibility layer maps
that API to `cudaMalloc`. There is no managed-memory, shared-memory or texture
allocation in these three kernels. Kernel bodies, input values and launch
mapping are unchanged by the verification repair.

## Workloads

Let `N = array_size_mb * 2^20 / sizeof(float)`, `B = block_size`,
`G = min(ceil(N/B), 65535)` and `S = G*B`. Input element `i` is
`float(i % 1000) * 0.001f`.

| Kernel | Work and output | Logical data bytes |
|---|---|---|
| `readOnly_kernel` | Each active thread sums `input[gid + k*S]` in FP32 and writes one result | Reads `4*N`; writes `4*min(N,S)` |
| `writeOnly_kernel` | Writes `float(i) * 0.001f` to every element | Writes `4*N` |
| `readWrite_kernel` | Multiplies each pristine input by `1.0001f` in place | Reads `4*N`; writes `4*N` |

These are logical accesses, not a claim about physical DRAM traffic. Cache
behavior and transaction sizes can differ. Derive bandwidth from the chosen
logical byte count and emitted kernel time; account for the read kernel's
output stores when reporting total logical traffic.

`transfer_mode=read` runs the read kernel, `write` runs the write kernel, and
`readwrite` runs all three in the order above. The third kernel's input is reset
with the existing recipe before its launch. The unused tail of the read-output
allocation is not a produced result and is never read by verification.

## Controls and build

```
make PLATFORM=cuda                    # or PLATFORM=rocm
./shoc_devicememory                   # 64 MiB, block 256, all three kernels
./shoc_devicememory --array_size_mb 96 --transfer_mode read
```

The flags `--array_size_mb`, `--block_size` and `--transfer_mode` have matching
`BENCH_PARAM_` variables, which take precedence. The first matching CLI flag is
used; all supplied values must be valid even when overridden. Numeric controls
must be positive 32-bit integers. Admission also checks the actual capped grid,
array byte size and final signed-index increment of the unchanged grid-stride
loops. The selected device's block and grid limits apply before allocation.
An integer-admissible manual input is not a resource-qualified configuration.

The declared profile remains 21 array sizes from 1 to 4096 MiB, block 256, and
three transfer modes: 63 configurations. `array_size_mb` retains its historical
name but means binary MiB. The warmer is a measurement control, not another
workload sweep dimension.

## Measurement and verification

The shared vector-add warmer runs immediately before each enabled timed kernel
unless `BENCH_PARAM_vector_add_warmup=false`. Setup, transfers, allocation and
verification are outside kernel timing. Each kernel emits one timing record;
the summary sums those recorded kernel times. Repetition remains the harness's
job (`gpubench run --reps N`).

Verification downloads each result in chunks of at most 1,048,576 floats (4 MiB).
Read results are checked before the write kernel overwrites them. Write and
read/write results are checked in full before their buffers are freed. With
the warmer disabled, these added downloads and CPU checks also affect the
inter-kernel delay/cache state; this mode has no controlled cache-entry claim.

The read input is periodic with period 1000. For stride `S`, each thread makes
`floor(N/S)` or one additional FP32 addition, with the same input-residue
sequence for each `gid % 1000`. The CPU caches both ordered sums for every
residue (2000 floats), then compares **every actual produced output**. This
exploits an exact property of the existing input; it does not sample outputs or
run a smaller GPU problem. Host tests independently enumerate all original loop
indices against materialized inputs to validate this reference.

The other references directly evaluate the existing write and multiply
expressions. Every actual and expected value must be finite and exactly equal
as an FP32 value. There is no tolerance or verification bypass. The first ten
mismatches per kernel are printed, all mismatches are counted, and any mismatch
returns nonzero after the enabled measurements finish. Missing setup or
verification resources also fail explicitly; the runner retains emitted timing
records with an error status.

GPU problem storage remains two `4*N` arrays. Peak explicit host storage is the
existing `4*N` input/reset buffer, a verification chunk up to 4 MiB and a read
reference of about 8 KiB, excluding runtime overhead. CPU reference storage
therefore does not grow by another full array.

Records include requested controls, actual element/grid/thread/output counts,
logical read/write bytes, `algorithm="local_grid_stride_v1"`,
`input="periodic_1000_fp32_v1"`, `allocation="device_malloc"`,
`precision="fp32"` and `verification="full_cpu_exact_v1"`.

See `../../tests/README.md` for host references, full-profile GPU/fault checks
and frozen-local-kernel comparisons. H100 qualification does not imply AMD,
trace/replay or other-host resource qualification.
