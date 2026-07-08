# polybench_atax

PolyBench ATAX benchmark: matrix-transpose times vector.

Computes **y = Aᵀ · (A · x)** where A is an NX×NY matrix and x, y are NY-vectors.

## Algorithm

Two GPU kernels are executed per iteration:

1. **atax_kernel1** — `tmp = A · x`  
   Each thread computes one row: `tmp[i] = Σⱼ A[i,j] * x[j]`

2. **atax_kernel2** — `y = Aᵀ · tmp`  
   Each thread computes one column: `y[j] = Σᵢ A[i,j] * tmp[i]`

## Metric

**GB/s** (memory bandwidth) — matrix A is read twice per pass (once per kernel):

```
GB/s = (2 * NX * NY * 4 bytes) / time_s / 1e9
```

Derived downstream from the emitted `time_ms` and params; not reported by the binary.

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`atax_kernel1`, `atax_kernel2`) with GPU event time and host time, then a
`summary` whose `total_time_ms` is the sum of all recorded GPU event times:

```
{"type":"kernel","name":"atax_kernel1","iter":0,"time_ms":1.234000,"host_time_ms":1.290000,"params":{"nx":4096,"ny":4096,"block_size":256}}
{"type":"kernel","name":"atax_kernel2","iter":0,"time_ms":2.345000,"host_time_ms":2.401000,"params":{"nx":4096,"ny":4096,"block_size":256}}
...
{"type":"summary","total_time_ms":25.053000}
```

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
# Default: NX=NY=4096
./polybench_atax

# Custom size
./polybench_atax --nx 2048 --ny 2048

# More recorded passes
BENCH_PARAM_num_iters=10 ./polybench_atax
```

## Parameters

| Flag / Env var | Default | Description |
|------|---------|-------------|
| `--nx NX` / `BENCH_PARAM_nx` | 4096 | Number of matrix rows |
| `--ny NY` / `BENCH_PARAM_ny` | 4096 | Number of matrix columns |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Platform Notes

### CUDA / ROCm (HIP)

- Source: `polybench_atax.hip`  
- Block size: 256 threads  
- Grid: `ceil(NX/256)` for kernel1, `ceil(NY/256)` for kernel2  
- Requires: `nvcc` (CUDA) or `hipcc` (ROCm)

### Metal (Apple Silicon / macOS)

- Source: `polybench_atax_metal.mm` + `polybench_atax.metal`  
- The `.metal` shader file must be in the same directory as the binary at runtime  
- Requires: macOS with Metal support, Xcode command-line tools

## Reference

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite, ATAX kernel.
