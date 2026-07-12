# tier2/heteromark_fir — FIR Filter (HeteroMark)

Benchmarks a 1D FIR (Finite Impulse Response) filter on the GPU. Each thread computes one output sample. Emits the raw kernel time of a single launch; throughput (GB/s) is derived downstream.

## Provenance & fidelity

Faithful reproduction of the [HeteroMark benchmark suite](https://github.com/NUCAR-DEV/Hetero-Mark) FIR workload.

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
| `heteromark_fir.hip` | HIP source — FIR kernel with shared memory, CPU reference, verification |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./heteromark_fir                              # defaults: N=1048576
./heteromark_fir --size 65536                 # fewer samples
```

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record (`iter:0`) carrying the GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"fir_filter_kernel","iter":0,"time_ms":3.210000,"host_time_ms":3.300000,"params":{"size":1048576,"num_taps":64,"block_size":256}}
{"type":"summary","total_time_ms":3.210000}
```

Derived metrics such as throughput (GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
FIR filter  |  Samples: 1048576  |  Taps: 64

fir_filter_kernel: gpu = 3.215 ms, host = 3.301 ms
Total GPU time: 3.215 ms
PASS
```

## Verification

- First 1024 output samples are compared against a CPU reference FIR implementation
- Tolerance: relative 1e-4 + absolute 1e-6
- Reports PASS/FAIL with mismatch details; on FAIL the process exits non-zero

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_fir.hip`; coefficients loaded into `__shared__` memory per block
