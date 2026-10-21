# tier2/heteromark_fir/cuda — FIR Filter (HeteroMark) (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

Benchmarks a 1D FIR (Finite Impulse Response) filter on the GPU. Each thread computes one output sample. Emits the raw kernel time of a single launch; throughput (GB/s) is derived downstream.

## Provenance & fidelity

An independent CUDA implementation of HeteroMark's causal FIR recurrence with
zero initial history. The reference sources are the [upstream CPU
implementation](https://github.com/NUCAR-DEV/Hetero-Mark/blob/aa685e02705f4be380cba215c6df10b42262a9cd/src/fir/fir_benchmark.cc)
and [CUDA
implementation](https://github.com/NUCAR-DEV/Hetero-Mark/blob/aa685e02705f4be380cba215c6df10b42262a9cd/src/fir/cuda/fir_cuda_benchmark.cu),
pinned at `aa685e02705f4be380cba215c6df10b42262a9cd`.

Upstream streams chunks and carries their input history between launches. This
version filters one contiguous input in a single launch, reading earlier samples
directly and staging coefficients in shared memory. The causal convolution and
its scaling with samples and taps are preserved; the launch and transfer layout
differ. The local input is `sin(0.01*i)+0.5`, with normalized `1/(k+1)`
coefficients, rather than upstream's time-seeded random input and coefficients.
These existing synthetic inputs remain unchanged by the verification repair.

## Algorithm

- **N** input samples (default: N=1,048,576 = 1M floats)
- **num_taps** filter coefficients (default: 64)
- 1D convolution: `output[i] = Σ coeff[k] × input[i−k]` for k = 0..num\_taps−1
- Each GPU thread computes one output sample
- Filter coefficients loaded into shared memory for fast reuse
- GB/s = (input\_size + output\_size) × sizeof(float) / time\_sec / 10⁹ (derived downstream from the emitted `time_ms` and params)

## Measurement

The FIR kernel launches **exactly once** per process execution (see the root
README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before the timed launch. Repetition =
`gpubench run --reps N`.

## Parameters

Environment variables override argv:

| Variable (or `--flag`) | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_size` / `--size` | Number of input samples | 1048576 |
| `BENCH_PARAM_num_taps` / `--num_taps` | Number of filter taps | 64 |
| `BENCH_PARAM_block_size` / `--block_size` | Threads per block | 256 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `heteromark_fir.cu` | CUDA source — FIR kernel with shared memory, CPU reference, verification |
| `Makefile` | nvcc build (`CUDA_ARCH`, default `native`) |

## Build

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

## Run

```bash
./heteromark_fir                              # defaults: N=1048576
./heteromark_fir --size 65536                 # fewer samples
```

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record (`iter:0`) carrying the GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"fir_filter_kernel","iter":0,"time_ms":3.210000,"host_time_ms":3.300000,"params":{"size":1048576,"num_taps":64,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":3.210000}
```

Derived metrics such as throughput (GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: NVIDIA H100 PCIe (id=0)
FIR filter  |  Samples: 1048576  |  Taps: 64

fir_filter_kernel: gpu = 3.215 ms, host = 3.301 ms
Total GPU time: 3.215 ms
FIR verification: PASS (1048576 samples, 0 mismatches)
```

## Verification

After timing, the CPU computes the complete causal FIR from the pristine input
and coefficients. Every actual GPU output sample is compared, including startup
history and the final partial block; no GPU rerun supplies the checked result.
The kernel record identifies this check as `verification: "full_cpu_v1"`.

The existing local acceptance criterion is retained: both values must be finite,
and `abs(gpu-reference) <= 1e-4*abs(reference)+1e-6`. This differs from upstream's
absolute `1e-5` check. Any mismatch returns nonzero. The diagnostic reports the
complete sample and mismatch counts, with at most ten individual examples.
Reference calculation, download and comparison remain outside kernel timing.

From the repository root, using new output directories:

```bash
# [TODO for cuda] these tests exercise ../hip/; not yet ported
python3 tests/fir_verification.py --host-only --work-dir /tmp/fir-host
python3 tests/fir_verification.py --platform cuda --work-dir /tmp/fir-gpu --sweep
```

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. The host check uses exact signed-coefficient and impulse-response examples with
fatal sanitizers. GPU cases cover startup and partial-block boundaries, all tap
counts, corruption before and after the former 1024-sample cutoff, nonfinite
outputs, both warmer settings and bounded diagnostics. `--sweep` additionally
runs all 108 declared configurations with the unmodified production binary and
warmer enabled, requiring complete verification within 60 seconds per process.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. H100 PCIe / CUDA 12.9 validation on 2026-09-28 passed all 108 configurations and
148 production/fault cases, with a maximum whole-process duration of 1.293
seconds while checking all 8,388,608 samples at 128 taps. Clang/GCC fatal host
sanitizers, four GPU tools, eight existing finite-comparison cases and six
native CLI/SQLite outcomes passed. The compiled FIR kernel instructions are
unchanged. These are correctness and observed process-budget results on a shared
GPU, not publication performance measurements.

## Platform Notes

- **CUDA**: compiled with `nvcc` from `heteromark_fir.cu`; coefficients loaded into `__shared__` memory per block
