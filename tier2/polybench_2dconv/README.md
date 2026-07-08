# tier2/polybench_2dconv — PolyBench 2D Convolution

GPU benchmark for PolyBench 2D Convolution. Applies a fixed 3×3 kernel to an
N×N input matrix. Emits raw per-iteration kernel times; throughput (GFLOPS)
is derived downstream.

## Algorithm

Each GPU thread computes one interior output element:

```
B[i][j] = c[0][0]*A[i-1][j-1] + c[0][1]*A[i-1][j] + c[0][2]*A[i-1][j+1]
         + c[1][0]*A[i  ][j-1] + c[1][1]*A[i  ][j] + c[1][2]*A[i  ][j+1]
         + c[2][0]*A[i+1][j-1] + c[2][1]*A[i+1][j] + c[2][2]*A[i+1][j+1]
         for 1 <= i < N-1, 1 <= j < N-1
```

Fixed PolyBench 3×3 kernel coefficients:

```
c = {{0.8, 0.2, 0.3},
     {0.2, 0.7, 0.4},
     {0.1, 0.2, 0.5}}
```

**GFLOPS** = `2 × 9 × (N−2)² / time_s / 1e9` (derived downstream from the
emitted times and params)

## Files

| File | Description |
|------|-------------|
| `polybench_2dconv.hip` | HIP kernel + host (CUDA/ROCm, self-contained) |
| `polybench_2dconv.metal` | Metal compute shader |
| `polybench_2dconv_metal.mm` | ObjC++ Metal host |
| `Makefile` | Platform auto-detect build |

## Build

### Auto-detect (recommended)

```bash
make
```

Platform is auto-detected:
- **Darwin** → `metal`
- **Linux + `/opt/rocm`** → `rocm`
- **Linux, no ROCm** → `cuda`

### Explicit platform

```bash
make PLATFORM=cuda    # NVIDIA (nvcc)
make PLATFORM=rocm    # AMD ROCm (hipcc)
make PLATFORM=metal   # Apple Metal (clang++)
```

## Run

```bash
./polybench_2dconv [--size N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--size N` | 2048 | Matrix dimension N×N |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `BENCH_PARAM_size` | 2048 | Matrix dimension N×N (overrides `--size`) |
| `BENCH_PARAM_block_size` | 256 | Reported thread-block size |
| `BENCH_PARAM_precision` | float | Precision label (`half`, `float`, `double`) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup launches (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed launches, each recorded |

## Output

**stdout** — JSON lines, one kernel record per timed iteration with the GPU
event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"convolution2D_kernel","iter":0,"time_ms":3.210000,"host_time_ms":3.270000,"params":{"size":2048,"block_size":256,"precision":"float"}}
...
{"type":"summary","total_time_ms":22.470000}
```

Derived metrics such as GFLOPS are computed downstream from `time_ms` and
the params.

**stderr**:
```
Device: NVIDIA A100 (id=0)
Matrix: 2048×2048

convolution2D_kernel iter 0: gpu = 3.210 ms, host = 3.270 ms
...
Total GPU time: 22.470 ms over 7 iterations
```

## Notes

- Border elements (row/col 0 and N−1) are skipped, matching the PolyBench reference.
- Input is initialized with `rand() % 100 / 10.0f` using seed 42 (reproducible).
- Metal: shader source is compiled at runtime from `polybench_2dconv.metal`
  (must reside alongside the binary).
- The benchmark runs `BENCH_PARAM_num_warmup` warmup launches (default 3)
  before the timed iterations.
