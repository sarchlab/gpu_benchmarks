# altis_cfd — Altis Euler solver on a synthetic tetrahedral mesh

Runs the pinned Altis `euler3d` finite-volume solver once: **10 timesteps**, each
with an old-state copy, a per-cell step-factor calculation and **three
Runge–Kutta flux/update stages**. Every final conserved variable and the final
old-state, flux and step-factor buffers are checked against a separate FP64 CPU
trajectory after timing.

## Provenance and input decision

The solver bodies come from Altis revision
[`042e2926d1fe9a5ea5b3e2d659b75bf925b152cc`](https://github.com/utcs-scea/altis/blob/042e2926d1fe9a5ea5b3e2d659b75bf925b152cc/src/cuda/level2/cfd/euler3d.cu).
See `cfd_kernels.h` and `UPSTREAM_LICENSE`. They retain the upstream FP32
arithmetic, four face-neighbor slots, artificial viscosity coefficient 0.2,
Mach 1.2 far field, slip-wall/far-field branches, step-factor expression and RK
update denominators 4, 3 and 2. The timestep count is the upstream constant 10.
The old-state device copy is an explicit timed kernel, as required by this
repository's measurement protocol. Copy and step-factor calculation run in one
stream; the upstream driver can overlap these independent operations on two
streams. No defining computation has been removed or moved to the CPU.

**The dataset is intentionally different from upstream.** The owner approved
this exception on 2026-09-28: use a valid, reproducible synthetic mesh and
nonuniform flow rather than the upstream random neighbor/normal recipe. That
recipe can produce out-of-range neighbor indices and nonphysical geometry;
its padded-normal initialization also leaves padding uninitialized. This port
is not a reproduction of the upstream random dataset or a particular wing
mesh. Its workload identity is `altis_euler3d_rk3_v1` with input
`tetrahedral_perturbed_flow_v1`. Results from the former local two-stage,
single-step Rusanov implementation are not equivalent measurements.

## Mesh and initial state

Each unit cube is split into six tetrahedra along its body diagonal, one for
each permutation of the three coordinate axes. Cubes are ordered x, then y,
then z, in a box with side `ceil(cuberoot(ceil(size/6)))`; the first `size`
tetrahedra are retained. This permits every declared size, including partial
cubes, without invalid neighbor references or overlapping cells.

Every cell has volume 1/6, used in the upstream `areas` slot. Face normals are
**inward area vectors**, consistent with the solver's positive flux update;
shared faces have reciprocal neighbors and opposite normals, and each cell's
four area vectors sum to zero. Exposed faces on y=0 are slip walls; all other
exposed faces are far-field boundaries. There are no invented random links.

For cube coordinates `(x,y,z)` and tetrahedron index `t` in `0..5`, define
`q = ((11*x + 7*y + 3*z + t) % 33 - 16) / 256`.
Initial density is `1.4 + q`, pressure is `1 + q/2`, and velocity is `(1.2,0,0)`.
The input stores FP32 conserved variables using the pinned host energy formula.
This nonuniform input exercises evolving fluxes and updates; it is not a
uniform-flow no-op. It uses no libc RNG. The recorded recipe and qualified build
identify the input; broader cross-host input qualification remains separate.

`padded_cells` rounds size up to a multiple of block size. Every padded geometry
and initial-state entry copies the last real cell, including **all** normal
planes. Padded cells participate in the upstream computation and verification;
all neighbor indices continue to refer only to real cells.

## Parameters and admission

| Parameter | Default | Meaning |
|---|---:|---|
| `BENCH_PARAM_size` / `--size` | 97000 | Real tetrahedral cells |
| `BENCH_PARAM_block_size` / `--block_size` | 256 | Threads per 1D block |
| `BENCH_PARAM_vector_add_warmup` | true | Shared entry warmer |

Environment values override command-line values. Positive decimal integers are
required. Block size must be in 1..1024 and fit the selected device; padded
normal-plane indexing must fit signed int. The program checks available device
memory against its buffers plus the enabled 768 MiB warmer. Allocation or
reference failure exits nonzero. These checks do not qualify every possible
manual size or host-memory budget.

## Timing and output

```sh
make PLATFORM=cuda                    # or PLATFORM=rocm
./altis_cfd --size 50000 --block_size 256
```

Initialization, allocation and uploads finish before the one entry warmer.
The full solver then runs without CPU verification, downloads or extra warmer
launches between stages, preserving the algorithm's intermediate data reuse.
Each kernel launch is timed and accumulated; output has four kernel records:

| Kernel | Launches per process | Cells processed |
|---|---:|---:|
| `copy_variables_kernel` | 10 | `10*padded_cells` |
| `compute_step_factor_kernel` | 10 | `10*padded_cells` |
| `compute_flux_kernel` | 30 | `30*padded_cells` |
| `rk_update_kernel` | 30 | `30*padded_cells` |

The summary is the sum of these GPU event times. Records include real/padded
cells, grid/block geometry, cube side, timestep/stage/launch counts, per-kernel
cell updates, buffer requirements, and algorithm/input/boundary/precision/
verification identities. Times are raw; rates are derived downstream. Repeated
measurements require fresh processes through the harness.

## Complete verification

A separate matrix-form CPU reference advances the same input through all ten
three-stage steps using FP64 arithmetic and the pinned rounded physical
constants. It does not invoke or reuse the production GPU flux functions.
After the measured run, it compares every element of:

- final density, three momenta and energy: `5*padded_cells` values;
- the last timestep's old state: `5*padded_cells` values;
- the last stage's flux: `5*padded_cells` values;
- the last timestep's step factors: `padded_cells` values.

Both values must be finite and satisfy
`abs(GPU - CPU) <= 5e-6 + 1e-5*abs(CPU)`.
This is a numerical comparison tolerance, not a bitwise cross-platform promise.
No sampling or verification bypass is provided. At most ten discrepancies are
printed, but the full comparison is completed; any mismatch exits nonzero.
Previously emitted timing records remain available and the harness marks the
run failed.

Device solver storage is `132*padded_cells` bytes, excluding the warmer.
Host input plus FP64 reference arrays require `216*padded_cells` bytes; checking
uses an additional transfer buffer of at most 4 MiB. Metadata's
`host_input_reference_bytes` includes these arrays and transfer buffer, not
allocator overhead, thread stacks or the runtime. Up to 16 host threads process
independent reference cells; unavailable worker creation falls back to the
remaining workers and caller. Verification is outside GPU timing but inside
the whole-process collection budget.

## Qualification tests

```sh
python3 tests/cfd_host.py --compiler g++       # from repository root
python3 tests/cfd_measured_result.py --platform cuda --work-dir /tmp/cfd-check
make -C tier2/altis_cfd PLATFORM=cuda test
```

Host tests reconstruct tetrahedral faces independently, verify reciprocal
geometry and all padding, compare complete trajectories against frozen upstream
FP32 scalar bodies, and check actual parameter admission under fatal ASan/UBSan.
The measured-result tests exercise every declared configuration, actual output
corruption in all four result buffers, padding and largest tails, omitted
stages, and unavailable setup/reference paths.

`bench_output_test --size N --block_size B` compares every copy, step-factor,
flux and update result with the frozen original CUDA bodies at every stage.
These qualification-only downloads are absent from production. CI compiles GPU
tests without claiming GPU execution. AMD hardware and trace/replay
qualification remain separate.
