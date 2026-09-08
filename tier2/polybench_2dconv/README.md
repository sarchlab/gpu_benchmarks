# tier2/polybench_2dconv — PolyBench 2D Convolution

GPU implementation of the pinned PolyBench/GPU 2D convolution operator. Applies a signed 3×3
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

Pinned 3×3 operator, written in row/column offset order:

```
c = {{ 0.2, 0.5, -0.8},
     {-0.3, 0.6, -0.9},
     { 0.4, 0.7,  0.1}}
```

The reference is [PolyBench-ACC at commit `70ea4ca`](https://github.com/cavazos-lab/PolyBench-ACC/blob/70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7/CUDA/stencils/convolution-2d/2DConvolution.cu),
whose CPU and GPU routines agree on this operator with `DATA_TYPE=float`. Its
coefficient names enumerate the first coordinate vertically: the first row
above is `c11, c21, c31`. This mapping matters for signed asymmetric inputs.

The source expression performs nine multiplications and eight additions per
interior output. A downstream source-operation model is
`GFLOPS = 17 × (N−2)² / time_s / 1e9`; generated instruction counts and physical
throughput require qualification on the target GPU.

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
- Output borders are initialized to zero outside timing and remain untouched by the kernel; the upstream comparison also checks only interior outputs.
- Independent 16×16 launch tiling is retained. The default/sweep sizes and current input generator remain suite configurations, not a claim of identical upstream setup.
- Input uses `rand() % 100 / 10.0f` after seed42, while the pinned setup uses `rand()/RAND_MAX`. Host-library RNG portability remains owner-deferred R-27.
- Main requires N≥3 and N*N to fit its signed kernel indices. The declared 22-size sweep is unchanged.
- Earlier versions used nine different coefficients. Recollect those performance results.

## Operator qualification

`make test PLATFORM=cuda` (or `rocm`) runs the actual GPU kernel against an
independent dense double reference. Nineteen cases include nine independent
3×3 impulse inputs, signed matrices, rectangular strides and partial tiles.
NaN-poisoned interiors catch missing writes; output borders and allocation
guards must remain untouched. Each unit impulse must return its exact float
coefficient. Other interior outputs use tolerance
`1e-5 * max(1, abs(reference))`, and non-finite results fail.

`bench_output_test --size N` checks every output at an individual declared size;
`--impulse 0` through `--impulse 8` isolate each coefficient. CI compiles this
test; execution requires a GPU. Ordinary timing runs still lack an independent
measured-result verifier (R-13). AMD execution remains R-19 and general
trace-region/artifact qualification remains R-23/R-24. Functional profiling
does not constitute performance collection.
