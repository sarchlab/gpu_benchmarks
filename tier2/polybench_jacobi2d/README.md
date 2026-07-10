# polybench_jacobi2d

PolyBench 2D Jacobi stencil benchmark — GPU implementation for CUDA, ROCm/HIP, and Apple Metal.

## Algorithm

Iterates a 2D Jacobi stencil on an N×N grid for TSTEPS time steps:

```
B[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] + A[i][j]) / 5.0
```

After each step, the source and destination buffers are swapped (double-buffer technique). Only interior points (`i = 1..N-2`, `j = 1..N-2`) are updated; boundary values remain fixed at 0.

## Files

| File | Description |
|------|-------------|
| `polybench_jacobi2d.hip` | HIP kernel + host code (CUDA and ROCm/HIP) |
| `polybench_jacobi2d.metal` | Metal compute shader |
| `polybench_jacobi2d_metal.mm` | Objective-C++ Metal host code |
| `Makefile` | Build system with platform auto-detection |
| `README.md` | This file |

## Build Instructions

### Auto-detect platform
```bash
make
```

### NVIDIA CUDA
```bash
make PLATFORM=cuda
```

### AMD ROCm/HIP
```bash
make PLATFORM=rocm
```

### Apple Metal
```bash
make PLATFORM=metal
```

### Clean
```bash
make clean
```

## Run Instructions

```bash
./polybench_jacobi2d [--n N] [--tsteps T]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--n N` | Grid size (N×N) | 1024 |
| `--tsteps T` | Number of time steps per pass | 50 |
| `--trials K` | Legacy trial count (echoed in params only; the number of timed passes is `BENCH_PARAM_iteration`) | 3 |

Environment variables override argv:

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_n` | Grid size (N×N) | 1024 |
| `BENCH_PARAM_tsteps` | Time steps per pass | 50 |
| `BENCH_PARAM_trials` | Legacy trial count (echoed in params) | 3 |
| `BENCH_PARAM_block_size` | Echoed in params | 256 |
| `BENCH_PARAM_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed passes, each recorded | 7 |

### Examples

```bash
# Default run (1024×1024, 50 steps per pass)
./polybench_jacobi2d

# Large grid
./polybench_jacobi2d --n 2048 --tsteps 100

# Quick test
./polybench_jacobi2d --n 512 --tsteps 10
```

## Output Format

**stdout** — JSON-lines: one `{"type":"kernel",...}` record per timed pass (one pass = TSTEPS kernel launches from the restored initial state) carrying the pass's accumulated GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"jacobi2d_kernel","iter":0,"time_ms":25.12,"host_time_ms":27.40,"params":{"n":1024,"tsteps":50,"trials":3,"block_size":256}}
{"type":"summary","total_time_ms":175.84}
```

**stderr** — device info, per-pass timing, and correctness check result:
```
Device: NVIDIA GeForce RTX 3080 (id=0)
Grid: 1024×1024  |  TSTEPS: 50  |  3 warmup + 7 timed passes

pass 0: gpu = 25.123 ms (50 steps)
...
Total GPU time: 175.840 ms over 7 passes
Correctness check (CPU vs GPU, 64×64, 50 steps): PASS (max_rel=0.00e+00)
```

## Performance Metric

Memory bandwidth is derived downstream from the emitted `time_ms` and params as:

```
GB/s = TSTEPS × 2 × N × N × sizeof(float) / time_s / 1e9
```

Each time step reads the full N×N source array and writes the full N×N destination array. The two grids ping-pong across the TSTEPS steps within a pass; the initial state is restored (untimed) before every pass, and the step count is fixed, so every pass measures identical work.

## Correctness Verification

A CPU reference computation is run on a small 64×64 grid for the same number of TSTEPS. The GPU result is compared against the CPU result. The test passes if the maximum relative error across all interior points is below 1e-3.

## Kernel Configuration

- Thread block size: 16×16
- Grid size: `ceil((N-2)/16) × ceil((N-2)/16)`
- Each thread computes one interior point per time step
