# tier2/polybench_3mm — Three Chained Matrix Multiplications (PolyBench 3mm)

Benchmarks three chained matrix multiplications using GPU compute:

```
E = A * B
F = C * D
G = E * F
```

All matrices are square N×N single-precision floats (default N=256). The 3mm pass runs exactly once per execution and emits one time record per kernel; GFLOPS is derived downstream.

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) 3mm workload.

## Algorithm

- Three sequential GPU kernel launches per execution
- Each kernel: one thread per output element, simple dot-product inner loop (no tiling)
- Default size: N=256 (all matrices 256×256)
- GFLOPS = (2·N³ + 2·N³ + 2·N³) / time\_sec / 1e9 = 6·N³ / time\_sec / 1e9 (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `polybench_3mm.hip` | HIP source — three matrix-multiply kernels + self-contained utilities |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./polybench_3mm                       # defaults: 256×256
./polybench_3mm --size 512
BENCH_PARAM_size=512 ./polybench_3mm
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 256 | Matrix dimension N |
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the timed pass |

The 3mm pass runs exactly once per execution; repetition is the harness's job (`gpubench run --reps N`).

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per
kernel (`mm3_kernel1`, `mm3_kernel2`, `mm3_kernel3`) with that kernel's GPU
event time (`time_ms`) and host wall-clock time (`host_time_ms`), followed by
a `{"type":"summary","total_time_ms":...}` line whose value is the sum of the
three GPU event times. Derived metrics (GFLOPS) are computed downstream.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 256×256 (all)

k1 gpu = 0.712 ms, k2 gpu = 0.708 ms, k3 gpu = 0.705 ms
Total GPU time: 2.125 ms
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `polybench_3mm.hip`

## Matrix Dimensions

With `--size N`:

| Matrix | Rows | Cols |
|--------|------|------|
| A      | N    | N    |
| B      | N    | N    |
| E=A*B  | N    | N    |
| C      | N    | N    |
| D      | N    | N    |
| F=C*D  | N    | N    |
| G=E*F  | N    | N    |
