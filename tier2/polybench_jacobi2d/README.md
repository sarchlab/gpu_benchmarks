# polybench_jacobi2d

PolyBench 2D Jacobi stencil benchmark — GPU implementation for CUDA and ROCm/HIP.

## Algorithm

Iterates a 2D Jacobi stencil on an N×N grid for TSTEPS time steps:

```
B[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] + A[i][j]) / 5.0
```

After each step, the source and destination buffers are swapped (double-buffer technique). Only interior points (`i = 1..N-2`, `j = 1..N-2`) are updated; boundary values remain fixed at 0.

**Divergence from canonical PolyBench:** each `tstep` here performs ONE stencil sweep followed by a buffer swap, whereas canonical PolyBench jacobi-2d performs TWO sweeps per tstep (A→B then B→A). `tsteps=T` here therefore corresponds to roughly T/2 canonical time steps.

## Files

| File | Description |
|------|-------------|
| `polybench_jacobi2d.hip` | HIP kernel + host code (CUDA and ROCm/HIP) |
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
| `--tsteps T` | Number of time steps per execution | 50 |

Environment variables override argv:

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_n` | Grid size (N×N) | 1024 |
| `BENCH_PARAM_tsteps` | Time steps per execution | 50 |
| `BENCH_PARAM_vector_add_warmup` | Vector-add warmer (clock boost + LLC eviction) before the timed pass | true |

The TSTEPS-step pass runs exactly once per execution; repetition is the harness's job (`gpubench run --reps N`).

### Examples

```bash
# Default run (1024×1024, 50 steps)
./polybench_jacobi2d

# Large grid
./polybench_jacobi2d --n 2048 --tsteps 100

# Quick test
./polybench_jacobi2d --n 512 --tsteps 10
```

## Output Format

**stdout** — JSON-lines: one `{"type":"kernel",...}` record for the single pass (TSTEPS kernel launches from the initial state) carrying the accumulated GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` equals that GPU event time:
```json
{"type":"kernel","name":"jacobi2d_kernel","iter":0,"time_ms":25.12,"host_time_ms":27.40,"params":{"n":1024,"tsteps":50}}
{"type":"summary","total_time_ms":25.12}
```

**stderr** — device info, timing, and correctness check result:
```
Device: NVIDIA GeForce RTX 3080 (id=0)
Grid: 1024×1024  |  TSTEPS: 50  |  single timed pass

gpu = 25.123 ms (50 steps)
Total GPU time: 25.123 ms over 1 pass
Correctness check (CPU vs GPU, 64×64, 50 steps): PASS (max_rel=0.00e+00)
```

## Performance Metric

Memory bandwidth is derived downstream from the emitted `time_ms` and params as:

```
GB/s = TSTEPS × 2 × N × N × sizeof(float) / time_s / 1e9
```

Each time step reads the full N×N source array and writes the full N×N destination array. The two grids ping-pong across the TSTEPS steps; the initial state is uploaded (untimed) before the pass, and the step count is fixed, so every execution measures identical work.

## Correctness Verification

A CPU reference computation is run on a small 64×64 grid for the same number of TSTEPS. The GPU result is compared against the CPU result. The test passes if the maximum relative error across all interior points is below 1e-3; on `FAIL` the process exits with a non-zero status.

## Kernel Configuration

- Thread block size: 16×16
- Grid size: `ceil((N-2)/16) × ceil((N-2)/16)`
- Each thread computes one interior point per time step
