# tier2/cuda_convolution_separable — Separable 2D Convolution

Benchmarks a **separable 2D convolution** (row pass + column pass) on a floating-point image. Emits per-iteration GPU/host kernel times; throughput (GB/s) is derived downstream.

Derived from the [CUDA SDK convolutionSeparable](https://github.com/NVIDIA/cuda-samples) sample.

## Algorithm

- **Image size**: W×H floats (default: 4096×4096)
- **Kernel radius**: R (default: 8, producing a 17-tap filter)
- **Gaussian kernel** weights generated from σ = R/3
- Two-pass separable convolution:
  1. **Row pass** — each thread loads a row segment (with halo) into shared memory, applies 1D horizontal convolution
  2. **Column pass** — each thread loads a column segment (with halo) into shared memory, applies 1D vertical convolution
- GB/s = 2 · W · H · sizeof(float) / time\_sec / 10⁹ (derived downstream from the emitted `time_ms` and params)

## Files

| File | Description |
|------|-------------|
| `cuda_convolution_separable.hip` | HIP source — row + column convolution kernels with shared memory |
| `cuda_convolution_separable.metal` | Metal compute shaders — row + column convolution kernels |
| `cuda_convolution_separable_metal.mm` | ObjC++ Metal host code |
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
./cuda_convolution_separable                                    # defaults: 4096x4096, radius=8
./cuda_convolution_separable --width 2048 --height 2048         # smaller image
./cuda_convolution_separable --radius 16                        # wider kernel
```

Environment variables override argv:

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_width` | Image width | 4096 |
| `BENCH_PARAM_height` | Image height | 4096 |
| `BENCH_PARAM_radius` | Filter kernel radius | 8 |
| `BENCH_PARAM_num_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed passes, each recorded | 7 |

## Output

**stdout** — JSON-lines: per timed pass, one `{"type":"kernel",...}` record per kernel (`convolution_row_kernel`, `convolution_col_kernel`) carrying that kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"convolution_row_kernel","iter":0,"time_ms":2.5,"host_time_ms":2.6,"params":{"width":4096,"height":4096,"radius":8}}
{"type":"kernel","name":"convolution_col_kernel","iter":0,"time_ms":2.6,"host_time_ms":2.7,"params":{"width":4096,"height":4096,"radius":8}}
{"type":"summary","total_time_ms":35.7}
```

Derived metrics such as throughput (GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Image: 4096x4096  |  Kernel radius: 8 (17-tap)  |  Iterations: 3 warmup + 7 timed

pass 0: row gpu = 2.500 ms, col gpu = 2.600 ms
...
Total GPU time: 35.700 ms over 7 passes
PASS
```

## Verification

- Compares GPU output against CPU separable convolution reference
- Checks a 512×512 sub-region (or full image if smaller) with relative tolerance

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_convolution_separable.hip`; uses constant memory for kernel weights, shared memory for row/column tiles
- **Metal**: compiled with `clang++` from `cuda_convolution_separable_metal.mm`; the Metal shader (`cuda_convolution_separable.metal`) is loaded and compiled at runtime from the same directory as the binary
