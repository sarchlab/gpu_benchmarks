# tier2/heteromark_fir — FIR Filter (HeteroMark)

Benchmarks a 1D FIR (Finite Impulse Response) filter on the GPU. Each thread computes one output sample. Emits per-iteration GPU/host kernel times; throughput (GB/s) is derived downstream.

Derived from the [HeteroMark benchmark suite](https://github.com/NUCAR-DEV/Hetero-Mark) FIR workload.

## Algorithm

- **N** input samples (default: N=1,048,576 = 1M floats)
- **NUM_TAPS** = 128 filter coefficients
- 1D convolution: `output[i] = Σ coeff[k] × input[i−k]` for k = 0..NUM\_TAPS−1
- Each GPU thread computes one output sample
- Filter coefficients loaded into shared memory (HIP) / threadgroup memory (Metal) for fast reuse
- GB/s = (input\_size + output\_size) × sizeof(float) / time\_sec / 10⁹ (derived downstream from the emitted `time_ms` and params)

## Files

| File | Description |
|------|-------------|
| `heteromark_fir.hip` | HIP source — FIR kernel with shared memory, CPU reference, verification |
| `heteromark_fir.metal` | Metal compute shader — FIR kernel with threadgroup memory |
| `heteromark_fir_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./heteromark_fir                              # defaults: N=1048576
./heteromark_fir --size 65536                 # fewer samples
```

Environment variables override argv:

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_size` | Number of input samples | 1048576 |
| `BENCH_PARAM_num_taps` | Number of filter taps | 64 |
| `BENCH_PARAM_block_size` | Threads per block | 256 |
| `BENCH_PARAM_warmup` | Warmup launches (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed launches, each recorded | 7 |

## Output

**stdout** — JSON-lines: one `{"type":"kernel",...}` record per timed iteration carrying the GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```json
{"type":"kernel","name":"fir_filter_kernel","iter":0,"time_ms":3.21,"host_time_ms":3.30,"params":{"size":1048576,"num_taps":64,"block_size":256}}
{"type":"summary","total_time_ms":22.47}
```

Derived metrics such as throughput (GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
FIR filter  |  Samples: 1048576  |  Taps: 128  |  Iterations: 3 warmup + 7 timed

fir_filter_kernel iter 0: gpu = 3.215 ms, host = 3.301 ms
...
Total GPU time: 22.470 ms over 7 iterations
PASS
```

## Verification

- First 1024 output samples are compared against a CPU reference FIR implementation
- Tolerance: relative 1e-4 + absolute 1e-6
- Reports PASS/FAIL with mismatch details

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `heteromark_fir.hip`; coefficients loaded into `__shared__` memory per block
- **Metal**: compiled with `clang++` from `heteromark_fir_metal.mm`; the Metal shader (`heteromark_fir.metal`) is loaded and compiled at runtime from the same directory as the binary; coefficients loaded into `threadgroup` memory
