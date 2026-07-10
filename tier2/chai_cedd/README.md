# tier2/chai_cedd — Canny Edge Detection (Chai)

Benchmarks Canny Edge Detection (CEDD) on the GPU using a 5-stage pipeline. Processes a synthetic grayscale image and reports per-kernel, per-pass GPU times; throughput (Mpixels/sec) is derived downstream.

Derived from the [Chai benchmark suite](https://github.com/chai-benchmarks/chai) CEDD workload.

## Algorithm

- **Input**: W×H grayscale image (default: 2048×2048, synthetic random)
- 5 GPU kernels launched sequentially:
  1. **Gaussian Blur** — 5×5 Gaussian smoothing to reduce noise
  2. **Sobel Gradient** — compute horizontal (Gx) and vertical (Gy) gradients
  3. **Magnitude + Direction** — compute gradient magnitude and quantize direction to 0°/45°/90°/135°
  4. **Non-Maximum Suppression** — thin edges by suppressing non-local-maxima along gradient direction
  5. **Hysteresis Thresholding** — double-threshold edge classification (low=50, high=100)
- Mpixels/sec = W × H / time\_sec / 10⁶ (derived downstream from the recorded times)

## Files

| File | Description |
|------|-------------|
| `chai_cedd.hip` | HIP source — 5 CEDD kernels, CPU reference, verification |
| `chai_cedd.metal` | Metal compute shaders — 5 CEDD kernels |
| `chai_cedd_metal.mm` | ObjC++ Metal host code |
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
| `BENCH_PARAM_warmup` | 3 | Warmup pipeline passes (untimed) |
| `BENCH_PARAM_iteration` | 7 | Timed pipeline passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record per pipeline kernel with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"gaussian_blur_kernel","iter":0,"time_ms":1.0241,"host_time_ms":1.0555,"params":{"width":2048,...}}
...
{"type":"summary","total_time_ms":35.9123}
```

Derived metrics (Mpixels/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Canny Edge Detection  |  Image: 2048x2048  |  Passes: 3 warmup + 7 timed

pass 0: gauss=1.024 sobel=0.512 magdir=0.480 nms=0.470 hyst=0.610 ms
...
Total GPU time: 35.912 ms over 7 passes
PASS (0.3% pixel mismatch in 122x122 interior, 45/14884)
```

## Verification

- A 128×128 sub-region is processed on both GPU and CPU
- Interior pixels (excluding 3-pixel border) are compared
- Reports PASS if mismatch rate < 5%, FAIL otherwise
- Small differences are expected due to border handling differences between full-image GPU and sub-image CPU

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `chai_cedd.hip`; Gaussian kernel in `__constant__` memory; 16×16 thread blocks
- **Metal**: compiled with `clang++` from `chai_cedd_metal.mm`; the Metal shader (`chai_cedd.metal`) is loaded and compiled at runtime; all 5 kernels dispatched in a single command buffer
