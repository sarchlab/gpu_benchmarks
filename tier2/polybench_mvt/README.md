# polybench_mvt

PolyBench MVT benchmark: matrix-vector product and transpose.

Computes:
- **x1 = A · y1** — matrix-vector product
- **x2 = Aᵀ · y2** — transpose matrix-vector product

where A is an N×N matrix and x1, x2, y1, y2 are N-vectors.

## Algorithm

Two GPU kernels are executed per iteration:

1. **mvt_kernel1** — `x1 = A · y1`  
   Each thread computes one row: `x1[i] = Σⱼ A[i,j] * y1[j]`

2. **mvt_kernel2** — `x2 = Aᵀ · y2`  
   Each thread computes one column: `x2[j] = Σᵢ A[i,j] * y2[i]`

## Metric

**GB/s** (memory bandwidth) — matrix A is read twice per pass (once per kernel).
The formula is applied downstream to the emitted per-iteration times:

```
GB/s = (2 * N * N * 4 bytes) / time_s / 1e9   (derived downstream)
```

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`mvt_kernel1`, `mvt_kernel2`) with that kernel's GPU event time and host
wall-clock time, then a `summary` whose `total_time_ms` is the sum of all
recorded GPU event times:

```json
{"type":"kernel","name":"mvt_kernel1","iter":0,"time_ms":6.123400,"host_time_ms":6.150000,"params":{"n":4096,"block_size":256,"precision":"float"}}
{"type":"kernel","name":"mvt_kernel2","iter":0,"time_ms":6.234500,"host_time_ms":6.260000,"params":{"n":4096,"block_size":256,"precision":"float"}}
...
{"type":"summary","total_time_ms":86.505300}
```

Derived metrics (e.g. GB/s) are computed downstream from `time_ms` and params.

## Build

```bash
# Auto-detect platform (Darwin → metal, /opt/rocm exists → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
# Default: N=4096
./polybench_mvt

# Custom size
./polybench_mvt --n 2048
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--n N` | 4096 | Matrix dimension (N×N) |

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_n` | Matrix dimension (N×N) | 4096 |
| `BENCH_PARAM_block_size` | Threads per block | 256 |
| `BENCH_PARAM_precision` | Precision label (half, float, double) | float |
| `BENCH_PARAM_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed passes, each recorded | 7 |

## Platform Notes

### CUDA / ROCm (HIP)

- Source: `polybench_mvt.hip`  
- Block size: 256 threads  
- Grid: `ceil(N/256)` for both kernels  
- Requires: `nvcc` (CUDA) or `hipcc` (ROCm)

### Metal (Apple Silicon / macOS)

- Source: `polybench_mvt_metal.mm` + `polybench_mvt.metal`  
- The `.metal` shader file must be in the same directory as the binary at runtime  
- Requires: macOS with Metal support, Xcode command-line tools

## Reference

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite, MVT kernel.
