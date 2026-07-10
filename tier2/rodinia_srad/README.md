# tier2/rodinia_srad — Speckle-Reducing Anisotropic Diffusion (Rodinia SRAD v2)

Iterative 2D image smoothing using SRAD (Perona-Malik-style anisotropic
diffusion). This is a **faithful port** of Rodinia SRAD v2: the device kernels
`srad_cuda_1` and `srad_cuda_2` are the original upstream kernels
(`rodinia/cuda/srad/srad_v2/srad_kernel.cu`), copied verbatim, including their
shared-memory neighborhood tiling and the compile-time `BLOCK_SIZE 16` thread
block. Only the host driver is restructured for the per-iteration timing
harness and JSON output.

Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/).

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
| `rodinia_srad.hip` | HIP/CUDA source — original `srad_cuda_1`/`srad_cuda_2` kernels + timing driver |
| `rodinia_srad.metal` | Metal compute shader (separate reimplementation, not the upstream kernel) |
| `rodinia_srad_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./rodinia_srad                                  # defaults: 512×512, 100 SRAD iters/pass, lambda 0.5
./rodinia_srad --image_size 1024
BENCH_PARAM_iteration=10 ./rodinia_srad --image_size 256 --num_iterations 200
./rodinia_srad --lambda 0.3
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_image_size` | 512 | Image dimension (N×N, rounded up to a multiple of 16) |
| `BENCH_PARAM_num_iterations` | 100 | SRAD iterations per pass (each runs both kernels) |
| `BENCH_PARAM_lambda` | 0.5 | Diffusion step size in (0,1) |
| `BENCH_PARAM_warmup` | 3 | Warmup passes (not recorded) |
| `BENCH_PARAM_iteration` | 7 | Timed passes, each recorded |

`block_size` is fixed at the original compile-time `BLOCK_SIZE=16` and is
emitted as a param for downstream derivation only.

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`srad_cuda_1`, `srad_cuda_2`) with that kernel's accumulated GPU event time
(`time_ms`) and host time (`host_time_ms`) across the pass's SRAD iterations,
then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"srad_cuda_1","iter":0,"time_ms":6.123,"host_time_ms":9.001,"params":{"image_size":512,"num_iterations":100,"block_size":16,"lambda":0.500}}
{"type":"kernel","name":"srad_cuda_2","iter":0,"time_ms":6.222,"host_time_ms":9.100,"params":{"image_size":512,"num_iterations":100,"block_size":16,"lambda":0.500}}
{"type":"summary","total_time_ms":86.415}
```

**stderr** — human-readable per-pass timing.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_srad.hip` (the faithful upstream kernels).
- **Metal**: a separate reimplementation in `rodinia_srad_metal.mm`; it is not the upstream kernel.
- The input image is generated synthetically (`J = exp(I)` for a random `I`) — no external input files required.
