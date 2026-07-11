# tier2/chai_cedd — Canny Edge Detection (Chai)

Benchmarks Canny Edge Detection (CEDD) on the GPU using a 5-stage pipeline. Processes a synthetic grayscale image exactly once per execution and reports per-kernel GPU times; throughput (Mpixels/sec) is derived downstream.

Derived from the [Chai benchmark suite](https://github.com/chai-benchmarks/chai) CEDD workload.

## Algorithm

- **Input**: W×H grayscale image (default: 2048×2048, synthetic random)
- 5 GPU kernels launched sequentially:
  1. **Gaussian Blur** — 5×5 Gaussian smoothing to reduce noise
  2. **Sobel Gradient** — compute horizontal (Gx) and vertical (Gy) gradients
  3. **Magnitude + Direction** — compute gradient magnitude and quantize direction to 0°/45°/90°/135°
  4. **Non-Maximum Suppression** — thin edges by suppressing non-local-maxima along gradient direction
  5. **Hysteresis Thresholding** — double-threshold edge classification (defaults: low=50, high=150). This is a **single-pass approximation**: a weak pixel (between the thresholds) is promoted only if one of its 8 neighbors is already a strong pixel — there is no iterative propagation of edge connectivity as in full Canny
- Mpixels/sec = W × H / time\_sec / 10⁶ (derived downstream from the recorded times)

## Files

| File | Description |
|------|-------------|
| `chai_cedd.hip` | HIP source — 5 CEDD kernels, CPU reference, verification |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./chai_cedd                       # defaults: W=2048, H=2048
./chai_cedd W=1024 H=1024         # smaller image
./chai_cedd W=4096 H=4096         # larger image
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_width` | 2048 | Image width |
| `BENCH_PARAM_height` | 2048 | Image height |
| `BENCH_PARAM_threshold_low` | 50 | Low hysteresis threshold |
| `BENCH_PARAM_threshold_high` | 150 | High hysteresis threshold |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed pipeline; set `false` to time as-is |

## Output

**stdout** — JSON lines: for the single timed pipeline run, one `{"type":"kernel",...}` record per pipeline kernel with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"gaussian_blur_kernel","iter":0,"time_ms":1.0241,"host_time_ms":1.0555,"params":{"width":2048,...}}
...
{"type":"summary","total_time_ms":3.0960}
```

Derived metrics (Mpixels/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Canny Edge Detection  |  Image: 2048x2048

gauss=1.024 sobel=0.512 magdir=0.480 nms=0.470 hyst=0.610 ms
Total GPU time: 3.096 ms
PASS (0.3% pixel mismatch in 116x116 interior, 39/13456)
```

## Verification

- A 128×128 sub-region is processed on both GPU and CPU, using the same
  low/high hysteresis thresholds as the GPU run
- Interior pixels (excluding a 6-pixel border, covering the pipeline's
  ~5-pixel influence radius) are compared
- Reports PASS if mismatch rate < 1%, FAIL otherwise (float rounding can
  still flip pixels sitting exactly at a threshold)
- A FAIL makes the benchmark exit non-zero (after all JSON output is emitted)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `chai_cedd.hip`; Gaussian kernel in `__constant__` memory; 16×16 thread blocks
