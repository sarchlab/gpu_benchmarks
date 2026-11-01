# mixbench — Tier 1 Microbenchmark (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

## Overview

`mixbench` varies a nominal compute-to-logical-byte ratio, inspired by
[github.com/ekondis/mixbench](https://github.com/ekondis/mixbench). It is a local
workload; the name does not establish equivalence to upstream implementations.

Each active thread loads one float (4 bytes), performs a dependent FP32 `fmaf`
chain, and stores one float (4 bytes). The logical byte count describes source
loads/stores; it does not establish traffic at a particular cache or memory level.
Each invocation runs one requested `BENCH_PARAM_flops_per_byte` ratio:

| Ratio (nominal FLOP/logical byte) | FMA calls/element | Nominal FLOPs/element |
|---|---:|---:|
| 0.5 | 2 | 4 |
| 2 | 8 | 16 |
| 4 | 16 | 32 |
| 8 | 32 | 64 |
| 32 | 128 | 256 |

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `mixbench.cu` | `nvcc` |

## What It Measures

The sweep measures how this dependent-chain workload responds to increasing
arithmetic work per element. Its bottleneck depends on the target, grid, compiler,
and memory behavior. A large buffer and a high ratio do not guarantee saturation
or theoretical FP32 peak throughput. Qualify compiled instructions and physical
traffic before making a hardware roofline claim; see
[the nominal intensity model](../../../docs/measurement_interpretation.md#mixbenchs-nominal-intensity-model).

## Build

The directory is **self-contained** — no CMake or external dependencies.

```bash
cd tier1/mixbench/cuda
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

```bash
# NVIDIA CUDA
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

### Clean

```bash
make clean
```

## Run

```bash
# Default 256 MiB array
./mixbench

# Custom array size (in MiB)
./mixbench 512
```

Each execution launches and times the kernel **exactly once** — there is no
in-program warmup or iteration loop. Unless `BENCH_PARAM_vector_add_warmup`
is set to `false` (default `true`), a vector-add warmer runs ~100 ms of
sustained load immediately before the timed launch, intended to raise clocks
and disturb cache state. Qualify the resulting state on the target GPU. Repeat measurements
come from fresh process re-executions: `gpubench run --reps N`.

## Example Output

Each invocation runs a single `flops_per_byte` ratio. Progress and device
info go to stderr; stdout carries JSON lines — one record for the single
timed launch plus a final summary. Illustrative timings:

```
{"type":"kernel","name":"mixbench_kernel","iter":0,"time_ms":0.380124,"host_time_ms":0.462201,"params":{"buffer_size_mb":256,"flops_per_byte":4.0,"threads_per_block":256}}
{"type":"summary","total_time_ms":0.380124}
```

- `time_ms` is the GPU event time for the single kernel launch;
  `host_time_ms` is the host bracket around event recording, launch submission,
  error checking, and synchronization.
- `total_time_ms` equals the single launch's GPU event time.
- GFLOPS is not printed; it is derived downstream from `time_ms` and the
  params in the record after work-count qualification. Sweep
  `flops_per_byte` across invocations to compare this workload at each ratio.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_buffer_size_mb` (or argv[1]) | 256 | Array size in MiB |
| `BENCH_PARAM_flops_per_byte` | 4.0 | Requested nominal FLOP/logical-byte ratio; see truncation below |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer before the timed launch; qualify clock/cache effects on the target |

## Operational Intensity Calculation

For each active element:

- Logical bytes: 4 bytes read + 4 bytes written = **8 bytes**.
- Nominal FLOPs: `num_fmas * 2` (one multiply and one add per FMA call).
- The host uses `num_fmas = max(1, int(flops_per_byte * 4))`.
- Nominal intensity: `num_fmas / 4` FLOP/logical byte. This equals the requested
  ratio for every declared sweep value; custom ratios may truncate or clamp.

For the whole array, `elements = buffer_size_mb * 1024 * 1024 / 4`, so nominal
GFLOP/s is `elements * num_fmas * 2 / (time_ms * 1e6)`. The current output rounds
the requested ratio to one decimal; use declared ratios for reproducible analysis
until effective override reporting is fixed under R-16 in
[the readiness tracker](https://github.com/sarchlab/gpu_benchmarks/issues/71).

## Notes

- The source stores each active element's final value; compiled work and
  physical traffic still need target qualification.
- The benchmark records one GPU event interval per process execution.