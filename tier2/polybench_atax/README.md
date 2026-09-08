# polybench_atax

PolyBench ATAX benchmark: matrix-transpose times vector.

Computes **y = Aᵀ · (A · x)** where A is an NX×NY matrix and x, y are NY-vectors.

## Provenance & fidelity

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite — the `atax` workload. Independent reimplementation, spot-verified to reproduce PolyBench's algorithm faithfully.

## Algorithm

The ATAX pass runs exactly once per execution, launching two GPU kernels:

1. **atax_kernel1** — `tmp = A · x`  
   Each thread computes one row: `tmp[i] = Σⱼ A[i,j] * x[j]`

2. **atax_kernel2** — `y = Aᵀ · tmp`  
   Each thread computes one column: `y[j] = Σᵢ A[i,j] * tmp[i]`

## Metric

**GB/s** (memory bandwidth) — matrix A is read twice (once per kernel):

```
GB/s = (2 * NX * NY * 4 bytes) / time_s / 1e9
```

Derived downstream from the emitted `time_ms` and params; not reported by the binary.

## Output

**stdout** — JSON lines: one `kernel` record per kernel
(`atax_kernel1`, `atax_kernel2`) with GPU event time and host time, then a
`summary` whose `total_time_ms` is the sum of the two GPU event times:

```
{"type":"kernel","name":"atax_kernel1","iter":0,"time_ms":1.234000,"host_time_ms":1.290000,"params":{"nx":4096,"ny":4096,"block_size":256}}
{"type":"kernel","name":"atax_kernel2","iter":0,"time_ms":2.345000,"host_time_ms":2.401000,"params":{"nx":4096,"ny":4096,"block_size":256}}
{"type":"summary","total_time_ms":3.579000}
```

**stderr** — device info, per-kernel timing, and a correctness check: the
first 8 elements of `y` are compared against a CPU reference. On mismatch
the check prints `FAIL` and the process exits with a non-zero status.

## Build

```bash
# Auto-detect platform (/opt/rocm exists → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
# Default: NX=NY=4096
./polybench_atax

# Custom size
./polybench_atax --nx 2048 --ny 2048
```

The pass runs exactly once per execution; repetition is the harness's job
(`gpubench run --reps N`).

## Parameters

| Flag / Env var | Default | Description |
|------|---------|-------------|
| `--nx NX` / `BENCH_PARAM_nx` | 4096 | Number of matrix rows |
| `--ny NY` / `BENCH_PARAM_ny` | 4096 | Number of matrix columns |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the timed pass |

## Platform Notes

### CUDA / ROCm (HIP)

- Source: `polybench_atax.hip`  
- Block size: 256 threads  
- Grid: `ceil(NX/256)` for kernel1, `ceil(NY/256)` for kernel2  
- Requires: `nvcc` (CUDA) or `hipcc` (ROCm)

## Reference

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite, ATAX kernel.

The fixed `verification` parameter is `prefix8_cpu_v1`: compare the first
`min(8, dimension)` elements of each existing output vector, with the same
float accumulation order, relative tolerance and non-finite rejection as
before. CPU references now compute only those final dot products and allocate
only the checked output prefix. This is a limited prefix check; it provides no
coverage guarantee for other output positions. Full-result qualification and
cross-device verification remain separate requirements.

ATAX still computes **all** of `tmp = A*x`, since every checked element of
`Aᵀ*tmp` depends on all rows. The reference cost is
O(nx×ny + nx×min(8,ny)), with nx temporary floats plus the checked prefix.

Input generation, GPU kernels, transfers and timing intervals are unchanged.
The CPU work remains outside GPU/host launch timing. The process still pays
for full input generation and allocation; measure its complete wall time when
admitting a configuration under a target's process budget.
