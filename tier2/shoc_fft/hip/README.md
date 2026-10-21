# tier2/shoc_fft/hip — Batched 512-point FFT (faithful SHOC FFT port)

Faithful port of the SHOC level1 **FFT** benchmark: **batched 512-point complex
FFTs**, one thread block per FFT, 64 threads/block, radix-8 decomposition (three
`FFT8` stages) with twiddle factors and shared-memory transposes. The device
kernels (`FFT512_device`, `IFFT512_device`) and all their codelet helpers are
the **original SHOC/Volkov code**, taken verbatim from
`shoc/src/cuda/level1/fft/codelets.h`, and driven with the forward-then-inverse
structure of `fftlib.cu` / `FFT.cpp` `runTest`. Only the host driver is
restructured for the single-run timing harness and JSON output.

## Provenance

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `FFT` workload. Independent reimplementation, spot-verified to reproduce SHOC's algorithm and constants faithfully: the Volkov radix-8 FFT and its twiddle-factor codelets are reproduced (taken verbatim from `shoc/src/cuda/level1/fft/codelets.h`). See "Deviations from upstream" below for the (host-only) differences in the verification path.

## Algorithm

- **n_ffts** independent 512-point complex FFTs, one CUDA/HIP block each
  (`grid2D(n_ffts)`, block = 64). Each thread owns 8 elements; three radix-8
  butterfly stages with intermediate twiddles and shared-memory transposes
  produce a 512-point transform (output in natural order).
- The measured run = the forward batched FFT (`FFT512_device`) over all
  `n_ffts` blocks, then the inverse batched FFT (`IFFT512_device`, which applies
  the built-in `1/512` normalization in-kernel, verbatim). Each is one launch
  and gets its own JSON record.
- GFLOPS (per SHOC) = `n_ffts · 5 · 512 · log2(512) / time_sec / 1e9`, derived
  downstream from the recorded times.

## Measurement protocol

The forward+inverse batched FFT runs **exactly once** per execution — there is
no in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup` is
`true` (default), the vector-add warmer runs after the (untimed) data upload
and immediately before the timed launches: ~100 ms of sustained load ramps the
GPU clock while its final pass evicts the last-level cache, so the kernels are
measured cold-cache at a boosted clock. Repetition is the harness's job:
`gpubench run --reps N` re-executes the binary in fresh processes.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_ffts K` | 32768 | Number of independent 512-point FFTs |
| `--precision P` | float | `float` (float2) or `double` (double2) kernel instantiation |

`fft_size` is fixed at 512 (compile-time in the SHOC kernels).

### Environment variables (override argv)

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_n_ffts` | 32768 | Number of 512-point FFTs |
| `BENCH_PARAM_precision` | float | `float` or `double` |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record for `FFT512_device`
and one for `IFFT512_device`, each with `time_ms` (GPU event time) and
`host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the
sum of the two recorded GPU event times:

```
{"type":"kernel","name":"FFT512_device","iter":0,"time_ms":...,"host_time_ms":...,"params":{"n_ffts":32768,"fft_size":512,"precision":"float"}}
{"type":"kernel","name":"IFFT512_device","iter":0,"time_ms":...,"host_time_ms":...,"params":{...}}
{"type":"summary","total_time_ms":...}
```

**stderr** — device info, per-kernel timing, and the round-trip correctness
check result.

## Verification

After the measured run (all JSON records and the summary already emitted), a
round-trip check verifies that forward then inverse (with the in-kernel `1/512`
normalization) recovers the original samples within `1e-2` max absolute error.
The process exits non-zero on FAIL, so the harness marks the run as failed
without losing the measurements. There is no verification on/off parameter.

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
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |
