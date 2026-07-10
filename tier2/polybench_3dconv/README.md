# tier2/polybench_3dconv — 3D Convolution (PolyBench)

Benchmarks 3D convolution over an **N×N×N** volume with a small 3D filter (e.g., 3×3×3). Each output voxel is the weighted sum of its spatial neighborhood. Emits per-iteration kernel times; compute throughput in GFLOPS is derived downstream.

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/).

## Algorithm

- 3D convolution: for each output point `(i,j,k)`, sum `input[i±r][j±r][k±r] * filter[fi][fj][fk]` where `r = filter_size/2`
- Default parameters: N=128, filter_size=3
- Boundary handling: zero-padding (out-of-bounds reads treated as 0)
- GFLOPS = 2 × N³ × filter_size³ / time_sec / 1e9 (derived downstream from the emitted times)

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` | Volume dimension N (N×N×N) | 128 |
| `BENCH_PARAM_block_size` | Threads per block dimension (block_size³ block) | 8 |
| `BENCH_PARAM_filter_size` | Filter dimension (filter_size³ kernel) | 3 |
| `BENCH_PARAM_warmup` | Warmup launches (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed launches, each recorded | 7 |

## Files

| File | Description |
|------|-------------|
| `polybench_3dconv.hip` | HIP/ROCm implementation |
| `polybench_3dconv_metal.mm` | ObjC++ Metal host code |
| `polybench_3dconv.metal` | Metal compute shader |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./polybench_3dconv                                      # defaults: 128³, filter 3³
BENCH_PARAM_size=256 ./polybench_3dconv                 # 256³ volume
BENCH_PARAM_filter_size=5 ./polybench_3dconv            # 5×5×5 filter
```

## Output

**stdout** — JSON lines, one `kernel` record per timed iteration followed by a
`summary` whose `total_time_ms` is the sum of all iterations' GPU event times:
```json
{"type":"kernel","name":"conv3d_kernel","iter":0,"time_ms":12.345678,"host_time_ms":12.401000,"params":{"size":128,"block_size":8,"filter_size":3}}
...
{"type":"summary","total_time_ms":86.419200}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Volume: 128x128x128  |  Filter: 3x3x3

conv3d_kernel iter 0: gpu = 12.346 ms, host = 12.401 ms
...
Total GPU time: 86.419 ms over 7 iterations
```

## Platform Notes

- **CUDA/ROCm**: 3D thread blocks (block_size³), compiled with `nvcc`/`hipcc`
- **Metal**: compiled with `clang++`; the Metal shader (`polybench_3dconv.metal`) is loaded at runtime from the same directory as the binary
