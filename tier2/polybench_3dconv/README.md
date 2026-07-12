# tier2/polybench_3dconv — 3D Convolution (PolyBench)

Benchmarks 3D convolution over an **N×N×N** volume with a small 3D filter (e.g., 3×3×3). Each output voxel is the weighted sum of its spatial neighborhood. The convolution runs exactly once per execution; the raw kernel time is emitted and compute throughput in GFLOPS is derived downstream.

## Provenance & fidelity

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) — the `3d-conv` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Generic runtime-filter convolution vs PolyBench's fixed sparse stencil; boundary semantics differ (random normalized filter with zero-padded borders, not the upstream fixed-coefficient stencil).

Read this as "a 3D-convolution benchmark in the spirit of PolyBench," not PolyBench's `3d-conv`.

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
| `BENCH_PARAM_vector_add_warmup` | Vector-add warmer (clock boost + LLC eviction) before the timed launch | true |

The kernel runs exactly once per execution; repetition is the harness's job (`gpubench run --reps N`).

## Files

| File | Description |
|------|-------------|
| `polybench_3dconv.hip` | HIP/ROCm implementation |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./polybench_3dconv                                      # defaults: 128³, filter 3³
BENCH_PARAM_size=256 ./polybench_3dconv                 # 256³ volume
BENCH_PARAM_filter_size=5 ./polybench_3dconv            # 5×5×5 filter
```

## Output

**stdout** — JSON lines, one `kernel` record for the single timed launch followed by a
`summary` whose `total_time_ms` equals that GPU event time:
```json
{"type":"kernel","name":"conv3d_kernel","iter":0,"time_ms":12.345678,"host_time_ms":12.401000,"params":{"size":128,"block_size":8,"filter_size":3}}
{"type":"summary","total_time_ms":12.345678}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Volume: 128x128x128  |  Filter: 3x3x3

conv3d_kernel: gpu = 12.346 ms, host = 12.401 ms
Total GPU time: 12.346 ms
```

## Platform Notes

- **CUDA/ROCm**: 3D thread blocks (block_size³), compiled with `nvcc`/`hipcc`
