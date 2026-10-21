# tier2/cuda_convolution_separable/hip — Separable 2D Convolution

Benchmarks a **separable 2D convolution** (row pass + column pass) on a floating-point image. Emits one GPU/host time record per kernel; throughput (GB/s) is derived downstream.

Derived from the [CUDA SDK convolutionSeparable](https://github.com/NVIDIA/cuda-samples) sample.

## Algorithm

- **Image size**: W×H floats (default: 4096×4096)
- **Kernel radius**: R (default: 8, producing a 17-tap filter; max 32)
- **Gaussian kernel** weights generated from σ = R/3
- Two-pass separable convolution:
  1. **Row pass** — each thread loads a row segment (with halo) into shared memory, applies 1D horizontal convolution
  2. **Column pass** — each thread loads a column segment (with halo) into shared memory, applies 1D vertical convolution
- GB/s = 2 · W · H · sizeof(float) / time\_sec / 10⁹ (derived downstream from the emitted `time_ms` and params)

## Measurement

The row+column kernel pair runs **exactly once** per process execution (see
the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

## Parameters

Environment variables override argv:

| Variable (or `--flag`) | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_width` / `--width` | Image width | 4096 |
| `BENCH_PARAM_height` / `--height` | Image height | 4096 |
| `BENCH_PARAM_radius` / `--radius` | Filter kernel radius (≤ 32) | 8 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `cuda_convolution_separable.hip` | HIP source — row + column convolution kernels with shared memory |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./cuda_convolution_separable                                    # defaults: 4096x4096, radius=8
./cuda_convolution_separable --width 2048 --height 2048         # smaller image
./cuda_convolution_separable --radius 16                        # wider kernel
```

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record (`iter:0`) per kernel (`convolution_row_kernel`, `convolution_col_kernel`) carrying that kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"convolution_row_kernel","iter":0,"time_ms":2.500000,"host_time_ms":2.600000,"params":{"width":4096,"height":4096,"radius":8,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"convolution_col_kernel","iter":0,"time_ms":2.600000,"host_time_ms":2.700000,"params":{"width":4096,"height":4096,"radius":8,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":5.100000}
```

Derived metrics such as throughput (GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Image: 4096x4096  |  Kernel radius: 8 (17-tap)

row gpu = 2.500 ms, col gpu = 2.600 ms
Total GPU time: 5.100 ms
PASS: 0 mismatches in 33554432 values (16777216 pixels per stage)
```

## Verification

Every pixel of both the actual measured row intermediate and final column output
is downloaded and checked after timing against the full CPU separable reference.
The reference uses the original ascending-tap FP32 arithmetic, zero padding and
Gaussian weights. Both values must be finite and satisfy
`abs(GPU - CPU) <= 1e-3 * abs(CPU) + 1e-5`. No GPU pass is rerun for verification.
Both kernel records carry `verification: "full_cpu_v1"`.

A mismatch or unavailable complete reference makes the process fail. All
mismatches are counted, with at most ten detailed diagnostics. Verification
retains the CPU row and column images and reuses one GPU-download host buffer;
peak host image storage remains four images (about 2 GiB at 16384×8192).
Reference computation, downloads and comparisons are outside the kernel timers,
but count toward the runner's whole-process timeout.

Regression commands from the repository root:

```bash
python3 tests/convolution_verification.py --host-only --work-dir /tmp/conv-host
python3 tests/convolution_verification.py --platform cuda --sweep --work-dir /tmp/conv-gpu
```

The host fixtures cover asymmetric coefficients, direct 2D known results,
zero-padding boundaries and immutable inputs. GPU regressions corrupt actual
post-timing device values in each stage, including points outside the old
512×512 region and NaNs, and exercise unavailable-reference failures.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_convolution_separable.hip`; uses constant memory for kernel weights, shared memory for row/column tiles
