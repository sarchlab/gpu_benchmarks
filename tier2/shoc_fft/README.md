# tier2/shoc_fft — Cooley-Tukey Radix-2 FFT (SHOC FFT)

Benchmarks a radix-2 iterative Cooley-Tukey FFT on **N complex elements** (stored as `float2`). Reports per-pass, per-kernel GPU times; GFLOPS is derived downstream.

Derived from the [SHOC benchmark suite](https://github.com/vetter/shoc) FFT workload.

## Algorithm

- **N** complex elements stored as `float2` (default: N=1,048,576 = 2²⁰)
- Iterative Cooley-Tukey radix-2 FFT:
  1. **Bit-reversal permutation** — reorder input elements by bit-reversed indices
  2. **log₂(N) butterfly stages** — each stage performs N/2 butterfly operations with twiddle factors
- Each butterfly: complex multiply + add/subtract = 10 FP ops
- GFLOPS = 5 · N · log₂(N) / time\_sec / 10⁹ (derived downstream from the recorded times)

## Files

| File | Description |
|------|-------------|
| `shoc_fft.hip` | HIP source — bit-reversal + butterfly kernels (one kernel per stage) |
| `shoc_fft.metal` | Metal compute shaders — bit-reversal + butterfly kernels |
| `shoc_fft_metal.mm` | ObjC++ Metal host code |
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
./shoc_fft                              # defaults: N=1048576
./shoc_fft --size 65536                 # smaller FFT
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 1048576 | FFT size (power of 2) |
| `BENCH_PARAM_precision` | float | Precision label (float, double, half; recorded in params) |
| `BENCH_PARAM_batch_count` | 1 | Batch count (recorded in params) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup FFT passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed FFT passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record for `bit_reverse_kernel` and one for `fft_butterfly_kernel` (accumulated over all log₂(N) stage launches), each with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"bit_reverse_kernel","iter":0,"time_ms":0.4123,"host_time_ms":0.4310,"params":{"size":1048576,...}}
{"type":"kernel","name":"fft_butterfly_kernel","iter":0,"time_ms":11.9333,"host_time_ms":12.6021,"params":{"size":1048576,...}}
...
{"type":"summary","total_time_ms":86.4192}
```

Derived metrics (GFLOPS) are computed downstream from `time_ms` and the params. The pristine input is re-uploaded (untimed) before every pass since the FFT transforms the buffer in place.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
FFT size: 1048576 (2^20)  |  Passes: 3 warmup + 7 timed

pass 0: bit_reverse gpu = 0.412 ms, butterfly gpu = 11.933 ms
...
Total GPU time: 86.419 ms over 7 passes
PASS (Parseval)
```

## Verification

- For small sizes (N ≤ 8192): element-wise comparison against CPU reference FFT
- For large sizes: Parseval's theorem check (energy conservation between time and frequency domains)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `shoc_fft.hip`; one kernel launch per butterfly stage
- **Metal**: compiled with `clang++` from `shoc_fft_metal.mm`; the Metal shader (`shoc_fft.metal`) is loaded and compiled at runtime from the same directory as the binary
