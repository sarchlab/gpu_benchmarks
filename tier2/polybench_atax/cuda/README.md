# polybench_atax (native CUDA)

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
{"type":"kernel","name":"atax_kernel1","iter":0,"time_ms":1.234000,"host_time_ms":1.290000,"params":{"nx":4096,"ny":4096,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"atax_kernel2","iter":0,"time_ms":2.345000,"host_time_ms":2.401000,"params":{"nx":4096,"ny":4096,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":3.579000}
```

**stderr** — device info, per-kernel timing, and a correctness check:
all elements of `y` are compared against a CPU reference. On mismatch
the check prints `FAIL` and the process exits with a non-zero status.

## Build

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
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

### CUDA

- Source: `polybench_atax.cu`  
- Block size: 256 threads  
- Grid: `ceil(NX/256)` for kernel1, `ceil(NY/256)` for kernel2  
- Requires: `nvcc` (CUDA)

## Reference

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite, ATAX kernel.

The fixed `verification` parameter is `full_cpu_v1`: compare every element
of the actual measured output vectors against the complete CPU formula after
timing. The original float accumulation order, relative-error threshold
(`1e-3`, with `abs(reference)+1e-6` in the denominator), and rejection of
non-finite GPU values, references and relative errors are retained. A mismatch
prints `FAIL` and exits nonzero. No separate GPU verification pass is launched.

The CPU reference performs both complete matrix-vector products, O(nx×ny) work,
with nx temporary floats and ny reference-output floats (160 KiB total at the
largest declared shape). All output vectors were already downloaded in full;
input generation, GPU kernels, transfers and timing intervals are unchanged. The
process pays for the complete CPU check outside GPU/host launch timing. Qualify
full process wall time and numerical tolerance on each target.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. `tests/matvec_measured_result.py` checks actual device-output corruption at
the first element, the old prefix boundary, the middle and the tail, including
finite errors and NaN/Inf. Run `python3 tests/matvec_measured_result.py
--platform cuda --work-dir /tmp/matvec-check` from the repository root with a
new, nonexistent output directory. CI compiles this GPU test
and runs `tests/matvec_reference_test.cpp` under host sanitizers. A timeout
fails the GPU test and retains completed logs and a partial report.
