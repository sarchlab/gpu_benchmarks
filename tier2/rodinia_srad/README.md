# tier2/rodinia_srad — Speckle-Reducing Anisotropic Diffusion (Rodinia SRAD)

Iterative 2D image smoothing using Perona-Malik anisotropic diffusion. Each pixel's value is updated based on local gradient statistics and a diffusion coefficient that suppresses smoothing at edges. Classic structured-grid benchmark with two kernels per iteration.

Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/) SRAD workload.

## Algorithm

For each pixel `(row, col)` in an `N×N` image, per iteration:

**Phase 1 — Compute gradients and diffusion coefficient:**
```
dN = J[row-1][col] - J[row][col]
dS = J[row+1][col] - J[row][col]
dW = J[row][col-1] - J[row][col]
dE = J[row][col+1] - J[row][col]

G2   = (dN² + dS² + dW² + dE²) / Jc²
L    = (dN + dS + dW + dE) / Jc
qsqr = (0.5·G2 - (1/16)·L²) / (1 + 0.25·L)²
c    = clamp(1 / (1 + (qsqr - q0sqr) / (q0sqr·(1 + q0sqr))), 0, 1)
```

**Phase 2 — Update image:**
```
D = c[N]·dN + c[S]·dS + c[W]·dW + c[E]·dE
J[row][col] += 0.25 · lambda · D
```

Boundary cells use clamped (no-flux) boundary conditions. `q0sqr = 0.05` (speckle noise variance), `lambda = 0.25` (diffusion step size).

## Files

| File | Description |
|------|-------------|
| `rodinia_srad.hip` | HIP source — two SRAD kernels + self-contained utilities |
| `rodinia_srad.metal` | Metal compute shader — `srad1_kernel` and `srad2_kernel` |
| `rodinia_srad_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./rodinia_srad                                            # defaults: 512×512, 50 SRAD iters/pass
./rodinia_srad --image_size 1024
BENCH_PARAM_num_iters=10 ./rodinia_srad --image_size 256 --num_iterations 100
./rodinia_srad --lambda 0.5
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_image_size` | 512 | Image dimension (N×N) |
| `BENCH_PARAM_num_iterations` | 50 | SRAD iterations per pass |
| `BENCH_PARAM_block_size` | 16 | Thread-block dimension (B×B) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (not recorded) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`srad1`, `srad2`) with that kernel's accumulated GPU event time (`time_ms`)
and host time (`host_time_ms`) across the pass's SRAD iterations, then a
summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"srad1","iter":0,"time_ms":6.123,"host_time_ms":7.001,"params":{"image_size":512,"num_iterations":50,"block_size":16}}
{"type":"kernel","name":"srad2","iter":0,"time_ms":6.222,"host_time_ms":7.100,"params":{"image_size":512,"num_iterations":50,"block_size":16}}
{"type":"summary","total_time_ms":86.415}
```
Effective bandwidth is derived downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: Apple M4 Pro
Image: 512×512  |  SRAD iters/launch: 50

pass 0: srad1 gpu = 6.123 ms, srad2 gpu = 6.222 ms
...
Total GPU time: 86.415 ms over 7 passes (50 SRAD iters/pass)
```

## Bandwidth Calculation (derived downstream)

Effective bandwidth accounts for all six device buffers accessed per SRAD iteration:
- **srad1**: reads `J` (1×), writes `dN`, `dS`, `dW`, `dE`, `c` (5×) → 6 arrays
- **srad2**: reads `J`, `dN`, `dS`, `dW`, `dE`, `c` (6×), writes `J` (1×) → 7 arrays
- **Total**: 13 × N² × sizeof(float) bytes per SRAD iteration

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_srad.hip`
- **Metal**: compiled with `clang++` from `rodinia_srad_metal.mm`; the Metal shader is loaded at runtime from the same directory as the binary
- Input image is generated synthetically (random floats in [0, 1]) — no external input files required
