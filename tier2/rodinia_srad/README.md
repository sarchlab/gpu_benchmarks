# tier2/rodinia_srad — Speckle-Reducing Anisotropic Diffusion (Rodinia SRAD v2)

Iterative 2D image smoothing using speckle-reducing anisotropic diffusion.
This is an adaptation of the Rodinia SRAD v2 shared-memory GPU kernels for
the suite's portable measurement driver.

## Provenance & fidelity

The source reference is Rodinia 3.1, commit
`366b283456506ef2fe2c2b7dc5e83e941ffc5524`:
[GPU kernels](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/srad/srad_v2/srad_kernel.cu),
[host driver and CPU implementation](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/srad/srad_v2/srad.cu), and
[block-size definitions](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/srad/srad_v2/srad.h).

The local kernels retain the arithmetic, shared-memory neighborhood tiling
and synchronization structure. Six halo loads use guarded index selection
instead of the pinned source's unconditional reads followed by boundary
overwrites. Those original reads can address outside the image; their behavior
is not guaranteed on either vendor. The local guards select the intended
clamped edge values before loading, including both opposite edges in a
single-block grid. This is an adapted port, not an unchanged source copy.

The local driver fixes `BLOCK_SIZE=16` (the upstream default), uses a square
image rounded up to a multiple of 16, and selects ROI rows/columns
`0..min(127,N-1)`. The pinned driver accepts rows, columns and ROI bounds as
arguments and rejects dimensions that are not multiples of 16. Both generate
the input with seed 7 and `J = exp(I)`. The local driver adds the entry warmer,
single-pass measurement and accumulated per-kernel JSON records. These
source comparisons do not establish full measured-result or AMD qualification;
those remain open in [the readiness tracker](https://github.com/sarchlab/gpu_benchmarks/issues/71).

## Algorithm

Each SRAD iteration runs two kernel launches over the `N×N` image, tiled into
`16×16` blocks:

- **`srad_cuda_1`** — loads each `16×16` tile plus its N/S/W/E halo rows/columns
  into shared memory (`temp`, `north`, `south`, `east`, `west`), computes the
  directional derivatives and the diffusion coefficient `c` (equ 33), and stores
  `c` and the four deltas.
- **`srad_cuda_2`** — reloads `c` (and its south/east halos) into shared memory,
  computes the divergence (equ 58), and updates the image `J` (equ 61).

The speckle coefficient `q0sqr` is recomputed **each iteration on the CPU** from
an image-statistics (`sum`/`sum2`) reduction over the ROI of the current image —
exactly as the original `srad.cu` host does — and passed to both kernels. This
host reduction, and the host↔device copies the original performs each iteration,
stay in the pass but **outside** the timed brackets.

## Files

| File | Description |
|------|-------------|
| `rodinia_srad.hip` | HIP/CUDA source — adapted `srad_cuda_1`/`srad_cuda_2` kernels + timing driver |
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
./rodinia_srad                                  # defaults: 512×512, 100 SRAD iterations, lambda 0.5
./rodinia_srad --image_size 1024
./rodinia_srad --image_size 256 --num_iterations 200
./rodinia_srad --lambda 0.3
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_image_size` | 512 | Image dimension (N×N, rounded up to a multiple of 16) |
| `BENCH_PARAM_num_iterations` | 100 | SRAD iterations (each runs both kernels) |
| `BENCH_PARAM_lambda` | 0.5 | Diffusion step size in (0,1) |
| `BENCH_PARAM_vector_add_warmup` | `true` | Cold-cache calibration before the timed run |

`block_size` is fixed at `BLOCK_SIZE=16` (the upstream default) and is
emitted as a param for downstream derivation only.

The full `num_iterations`-step diffusion runs **exactly once**; repetition is
the harness's job (`gpubench run --reps N`).

## Output

**stdout** — JSON lines: one `kernel` record per kernel
(`srad_cuda_1`, `srad_cuda_2`) with that kernel's accumulated GPU event time
(`time_ms`) and host time (`host_time_ms`) across all SRAD iterations,
then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"srad_cuda_1","iter":0,"time_ms":6.123,"host_time_ms":9.001,"params":{"image_size":512,"num_iterations":100,"block_size":16,"lambda":0.500}}
{"type":"kernel","name":"srad_cuda_2","iter":0,"time_ms":6.222,"host_time_ms":9.100,"params":{"image_size":512,"num_iterations":100,"block_size":16,"lambda":0.500}}
{"type":"summary","total_time_ms":12.345}
```

**stderr** — human-readable timing.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_srad.hip` (the adapted SRAD v2 kernels).
- The input image is generated synthetically (`J = exp(I)` for a random `I`) — no external input files required.
