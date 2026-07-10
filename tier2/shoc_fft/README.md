# tier2/shoc_fft — Batched 512-point FFT (faithful SHOC FFT port)

Faithful port of the SHOC level1 **FFT** benchmark: **batched 512-point complex
FFTs**, one thread block per FFT, 64 threads/block, radix-8 decomposition (three
`FFT8` stages) with twiddle factors and shared-memory transposes. The device
kernels (`FFT512_device`, `IFFT512_device`) and all their codelet helpers are
the **original SHOC/Volkov code**, taken verbatim from
`shoc/src/cuda/level1/fft/codelets.h`, and driven with the forward-then-inverse
structure of `fftlib.cu` / `FFT.cpp` `runTest`. Only the host driver is
restructured for the per-iteration timing harness and JSON output.

## Algorithm

- **n_ffts** independent 512-point complex FFTs, one CUDA/HIP block each
  (`grid2D(n_ffts)`, block = 64). Each thread owns 8 elements; three radix-8
  butterfly stages with intermediate twiddles and shared-memory transposes
  produce a 512-point transform (output in natural order).
- One recorded pass = the forward batched FFT (`FFT512_device`) over all
  `n_ffts` blocks, then the inverse batched FFT (`IFFT512_device`, which applies
  the built-in `1/512` normalization in-kernel, verbatim). Each is one launch
  per pass and gets its own JSON record.
- GFLOPS (per SHOC) = `n_ffts · 5 · 512 · log2(512) / time_sec / 1e9`, derived
  downstream from the recorded times.

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_ffts K` | 32768 | Number of independent 512-point FFTs |
| `--precision P` | float | `float` (float2) or `double` (double2) kernel instantiation |

`fft_size` is fixed at 512 (compile-time in the SHOC kernels).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_n_ffts` | 32768 | Number of 512-point FFTs |
| `BENCH_PARAM_precision` | float | `float` or `double` |
| `BENCH_PARAM_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_iteration` | 7 | Timed passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record for
`FFT512_device` and one for `IFFT512_device`, each with `time_ms` (GPU event
time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms`
is the sum of all recorded GPU event times:

```
{"type":"kernel","name":"FFT512_device","iter":0,"time_ms":...,"host_time_ms":...,"params":{"n_ffts":32768,"fft_size":512,"precision":"float"}}
{"type":"kernel","name":"IFFT512_device","iter":0,"time_ms":...,"host_time_ms":...,"params":{...}}
...
{"type":"summary","total_time_ms":...}
```

**stderr** — device info, per-pass timing, and a round-trip correctness check.

## Data reset & verification

The transform is data-independent in time, but forward+inverse mutates the
buffer (and rounding accumulates), so the pristine source is re-uploaded
(untimed) at the start of every pass. Correctness is checked once at the end via
round-trip error: forward then inverse (with the in-kernel `1/512`
normalization) must recover the original samples.

## Deviations from upstream

- Verification uses a host round-trip comparison instead of SHOC's
  `chk512_device` self-comparison kernel; the doubled work buffer that check
  required is therefore not allocated. The core compute kernels are unchanged.
- Precision is `float`/`double` (the `chk512`/`norm512`/cuFFT paths are not
  ported). The twiddle codelets use `__cosf`/`__sinf` exactly as upstream.

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./shoc_fft                        # defaults: n_ffts=32768, float
./shoc_fft --n_ffts 65536
BENCH_PARAM_precision=double ./shoc_fft
```

## Files

| File | Description |
|------|-------------|
| `shoc_fft.hip` | HIP/CUDA source — verbatim SHOC FFT512/IFFT512 codelets + timing driver |
| `shoc_fft.metal` | Metal compute shaders (legacy radix-2 implementation) |
| `shoc_fft_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
