# tier2/rodinia_lavamd — Short-Range Molecular Dynamics (Rodinia LavaMD)

Short-range molecular dynamics with cell-list decomposition. This is a
**faithful port** of Rodinia LavaMD: the device kernel `kernel_gpu_cuda` is the
original upstream kernel (`rodinia/cuda/lavaMD/kernel/kernel_gpu_cuda.cu`),
copied verbatim, including its staging of the home box and each neighbor box's
particles into shared memory before the all-pairs interaction. Only the host
driver is restructured for the single-pass timing harness and JSON output.

## Provenance & fidelity

Derived from the [Rodinia benchmark suite](http://lava.cs.virginia.edu/Rodinia/) — the `lavaMD` workload. Independent reimplementation, spot-verified to reproduce Rodinia's algorithm faithfully.

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
{"type":"kernel","name":"kernel_gpu_cuda","iter":0,"time_ms":12.345,"host_time_ms":12.501,"params":{"boxes1d":10,"particles_per_box":100,"number_threads":128,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":12.345}
```

The benchmark launches `kernel_gpu_cuda` over all home boxes **exactly once**
(as the original does); repetition is the harness's job
(`gpubench run --reps N`). Because the kernel *accumulates* into the force
output, that output buffer is zeroed before the launch (untimed); the
immutable position/charge/box inputs need no reset.

**stderr** — human-readable timing.

## Measured-result verification

Every ordinary run checks all four energy/force components (`v`, `x`, `y`, `z`)
of every actual measured particle after GPU timing. The independent CPU
reference evaluates the complete double-precision pair sums from the original
positions and charges. It includes self interactions and the home box plus
valid adjacent boxes; CPU neighbors are derived from grid coordinates rather
than copied from the neighbor list supplied to the GPU.

The reference follows the energy and force formulas in the pinned
[Rodinia 3.1 CPU kernel](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/openmp/lavaMD/kernel/kernel_cpu.c).
A component passes only when actual, reference and error are finite and
`abs(actual - reference) <= 1e-10 * max(1, abs(reference))`. The absolute floor
handles force components near zero; errors within this stated tolerance are
accepted. Qualification checks independent long-double sums, exact uniform
geometry cases, zero/signed charges and measured-output faults. AMD execution
and other-device tolerance qualification remain separate requirements.

The checker counts every mismatch, prints at most ten details to stderr and
exits nonzero on failure. The kernel record includes
`verification="full_cpu_v1"`. There is no bypass or separate GPU verification
launch. After the reference consumes the original positions, their host buffer
is reused for one complete device-to-host copy of the measured output. The
GPU kernel, input generation, initial output memset, launch geometry, warmer
and timed event sequence are unchanged.

At the largest declared 20³ boxes, there are 800,000 particles and 3,200,000
checked components. The reference adds a 25,600,000-byte result buffer (about
24.4 MiB); the equally sized result download reuses existing host storage.
Full reference work grows with every box-particle interaction and adds CPU
process time outside GPU timing. Host admission, other-host process budgets
and full upstream-fidelity qualification remain open.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_lavamd.hip` (the faithful upstream kernel).
- Input data is generated synthetically — no external input files required.
