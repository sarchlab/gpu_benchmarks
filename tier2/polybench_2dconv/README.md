# tier2/polybench_2dconv — PolyBench 2D Convolution

GPU benchmark modeled on PolyBench/GPU 2D Convolution. Applies a fixed 3×3
kernel to an N×N input matrix, running the convolution exactly once per
execution. Emits the raw kernel time; throughput (GFLOPS) is derived
downstream.

## Algorithm

Each GPU thread computes one interior output element:

```
B[i][j] = c[0][0]*A[i-1][j-1] + c[0][1]*A[i-1][j] + c[0][2]*A[i-1][j+1]
         + c[1][0]*A[i  ][j-1] + c[1][1]*A[i  ][j] + c[1][2]*A[i  ][j+1]
         + c[2][0]*A[i+1][j-1] + c[2][1]*A[i+1][j] + c[2][2]*A[i+1][j+1]
         for 1 <= i < N-1, 1 <= j < N-1
```

Fixed 3×3 coefficient set (same 9-tap structure as PolyBench/GPU, but the
values differ from upstream's signed c11..c33 set):

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
| `Makefile` | Platform auto-detect build |

## Build

### Auto-detect (recommended)

```bash
make
```

Platform is auto-detected:
- **Linux + `/opt/rocm`** → `rocm`
- **Linux, no ROCm** → `cuda`

### Explicit platform

```bash
make PLATFORM=cuda    # NVIDIA (nvcc)
make PLATFORM=rocm    # AMD ROCm (hipcc)
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
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer (clock boost + LLC eviction) before the timed launch |

The kernel runs exactly once per execution; repetition is the harness's job
(`gpubench run --reps N`).

## Output

**stdout** — JSON lines: one kernel record for the single timed launch with
the GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`),
then a summary whose `total_time_ms` equals that GPU event time:
```
{"type":"kernel","name":"convolution2D_kernel","iter":0,"time_ms":3.210000,"host_time_ms":3.270000,"params":{"size":2048}}
{"type":"summary","total_time_ms":3.210000}
```

Derived metrics such as GFLOPS are computed downstream from `time_ms` and
the params.

**stderr**:
```
Device: NVIDIA A100 (id=0)
Matrix: 2048×2048

convolution2D_kernel: gpu = 3.210 ms, host = 3.270 ms
Total GPU time: 3.210 ms
```

## Notes

- Border elements (row/col 0 and N−1) are skipped, matching the PolyBench reference structure.
- The coefficient values are a fixed set specific to this implementation; canonical PolyBench/GPU uses a different (signed) c11..c33 set. The 9-tap access pattern and FLOP count are the same.
- Input is initialized with `rand() % 100 / 10.0f` using seed 42 (reproducible).
