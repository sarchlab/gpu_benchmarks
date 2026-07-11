# tier2/rodinia_lavamd — Short-Range Molecular Dynamics (Rodinia LavaMD)

Short-range molecular dynamics with cell-list decomposition. This is a
**faithful port** of Rodinia LavaMD: the device kernel `kernel_gpu_cuda` is the
original upstream kernel (`rodinia/cuda/lavaMD/kernel/kernel_gpu_cuda.cu`),
copied verbatim, including its staging of the home box and each neighbor box's
particles into shared memory before the all-pairs interaction. Only the host
driver is restructured for the single-pass timing harness and JSON output.

Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/).

## Algorithm

Space is a `boxes1d³` grid of boxes; each box holds `NUMBER_PAR_PER_BOX = 100`
particles. One thread block processes one home box:

```
a2 = 2 * alpha^2
load home-box particles rA into shared memory (rA_shared)
for each of the (up to 26) neighbor boxes + self:
    load neighbor particles rB and charges qB into shared memory
    for each home particle i:
        for each neighbor particle j:
            r2  = rA.v + rB.v - dot(rA, rB)
            vij = exp(-a2 * r2)          // modified-LJ / Gaussian potential
            fs  = 2 * vij                // as in upstream (a2 does not appear in fs)
            fA[i] += qB[j] * (vij, fs*dx, fs*dy, fs*dz)
```

Note this is the original modified-Lennard-Jones / Gaussian potential
(`vij = exp(-a2*r2)`), **not** a standard `1/r^12 − 1/r^6` Lennard-Jones form.

## Files

| File | Description |
|------|-------------|
| `rodinia_lavamd.hip` | HIP/CUDA source — original `kernel_gpu_cuda` + timing driver |
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
./rodinia_lavamd                       # default: 10³ boxes, 100 particles/box
./rodinia_lavamd --boxes1d 15
BENCH_PARAM_boxes1d=8 ./rodinia_lavamd
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_boxes1d` | 10 | Boxes per dimension (total boxes = `boxes1d³`) |
| `BENCH_PARAM_vector_add_warmup` | `true` | Cold-cache calibration before the timed launch |

`particles_per_box` (100) and `number_threads` (128) are fixed compile-time
constants (`NUMBER_PAR_PER_BOX`, `NUMBER_THREADS`) in the original kernel and are
emitted as params for downstream derivation only.

## Output

**stdout** — JSON lines: one `kernel` record for
`kernel_gpu_cuda` carrying the GPU event time (`time_ms`) and host wall-clock
time (`host_time_ms`), then a summary whose `total_time_ms` is the recorded
GPU event time:
```json
{"type":"kernel","name":"kernel_gpu_cuda","iter":0,"time_ms":12.345,"host_time_ms":12.501,"params":{"boxes1d":10,"particles_per_box":100,"number_threads":128}}
{"type":"summary","total_time_ms":12.345}
```

The benchmark launches `kernel_gpu_cuda` over all home boxes **exactly once**
(as the original does); repetition is the harness's job
(`gpubench run --reps N`). Because the kernel *accumulates* into the force
output, that output buffer is zeroed before the launch (untimed); the
immutable position/charge/box inputs need no reset.

**stderr** — human-readable timing.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_lavamd.hip` (the faithful upstream kernel).
- Input data is generated synthetically — no external input files required.
