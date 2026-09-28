# tier2/cuda_nbody — All-Pairs N-Body Simulation (CUDA N-Body)

Benchmarks a gravitational N-body simulation using the **classic tile-based shared memory optimization**. Each thread computes the force on one body from all others. Emits the run's kernel time; throughput in GFLOPS is derived downstream.

## Provenance & fidelity

Faithful reproduction of the [CUDA N-Body sample](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda): the classic GPU Gems 3 tile-based all-pairs N-body kernel is reproduced. (Now double-buffers positions.)

## Algorithm

- **N** bodies (default: 16,384), 10 timesteps per run
- All-pairs gravitational force computation with O(N²) interactions
- **Tile-based shared memory optimization**: bodies loaded in tiles of 256 (compile-time `TILE_SIZE`, also the block size) into shared memory for data reuse
- Each interaction: distance computation, softening (ε=1e-5), inverse cube law → ~20 FLOPs
- After force accumulation: velocity update followed by position integration (dt=0.01)
- GFLOPS = 20 · N² · timesteps / time\_sec / 10⁹ (derived downstream from the emitted times)
- This is a logical-work estimate. Partial tiles also execute zero-mass padding interactions; each active body executes `ceil(N/256)*256` interaction slots per timestep. Integration and padding are not included in the 20·N² rate, which is not an instruction-count measurement.
- Particle positions/velocities are uploaded pristine before the run (untimed); the run then integrates all `timesteps`

## Measurement

The simulation (all `timesteps` kernel launches) runs **exactly once** per
process execution (see the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_bodies` / `--bodies` | Number of bodies | 16384 |
| `BENCH_PARAM_timesteps` / `--timesteps` | Simulation timesteps per run | 10 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `cuda_nbody.hip` | HIP source — tile-based N-body kernel with shared memory |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./cuda_nbody                                   # defaults: N=16384, 10 timesteps
./cuda_nbody --bodies 8192                     # fewer bodies
./cuda_nbody --bodies 16384 --timesteps 20
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) with the GPU event
time accumulated over all timesteps of the single run, followed by a `summary`
whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"nbody_kernel","iter":0,"time_ms":234.567800,"host_time_ms":235.100000,"params":{"bodies":16384,"timesteps":10,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":234.567800}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: NVIDIA A100 (id=0)
Bodies: 16384  |  Timesteps: 10

nbody_kernel gpu = 234.568 ms, host = 235.100 ms
Total GPU time: 234.568 ms
PASS (verified 16384 bodies, 10 timesteps, positions and velocities, 0 mismatches)
```

## Verification

After timing, the driver downloads the actual final position buffer and all
velocities. The CPU reference starts from the pristine input and performs all
requested timesteps on all bodies. Both final XYZ vectors are checked using
independent tolerances of the same form:

```text
tolerance = 0.01 * (abs(reference.x) + abs(reference.y) + abs(reference.z)) + 0.00001
abs(actual.component - reference.component) <= tolerance, for each XYZ component
```

All actual/reference components and each tolerance must be finite. Position
mass (`w`) and velocity padding (`w`) must equal their unchanged reference values.
The comparison checks every body, caps detailed diagnostics at ten mismatching
bodies, and reports the full body/timestep count and mismatch count. A mismatch
or unavailable CPU reference returns failure, including when timed records were
already emitted. Records identify `verification: "full_cpu_v1"`; a policy label
alone does not establish a successful check.

The GPU kernel, initial state, integration, timestep size, timing and original
ordered CPU recurrence are preserved. The former separate one-step GPU subset
check is removed. CPU verification runs after timing and is included in the
harness's **60-second process budget**. Host storage is approximately 76 bytes
per body (four `float4` state arrays and one `float3` acceleration array); GPU
state uses 48 bytes per body, in addition to runtime and warmer allocations.

The GPU uses `rsqrtf` and can fuse arithmetic; the CPU uses `1/sqrtf` in its
existing ordered recurrence. Some complete trajectories disagree beyond the
stated tolerance. Such a result is unsupported for this collection profile;
it is not, by itself, proof that the GPU trajectory is mathematically wrong.
Neither the tolerance nor the integration is changed to make those cases pass.

## Initial H100 collection profile

The general manifest retains all 75 body-count/timestep combinations. It is
**not** the initial qualified collection list. The complete classification in
[h100_profile.json](h100_profile.json) accounts for every declared combination,
records the qualified binary/source hashes and target/compiler, and separates
observed trajectory mismatches from configurations not yet qualified.

The initial assessment tests combinations with at most `2^32` CPU pair
interactions (`bodies * bodies * timesteps`). This bounds assessment work; it
is not a benchmark input limit or a claim about untested runtime. All assessed
processes still have a 60-second timeout. The initial H100 PCIe results admit:

| Timesteps | Qualified body counts |
|---|---|
| 1 | 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536 |
| 10 | 256, 384, 2048, 4096, 12288, 16384 |
| 50 | None in the initial profile |

Of the 41 assessed combinations, 23 pass complete position/velocity checks
and 18 disagree with the CPU trajectory. The remaining 34 are outside the
initial assessment scope. All 52 excluded combinations remain explicitly
unsupported pending further work; they must not contribute performance results.
The default 16384/10 is included. These are single-process qualification results,
not publication performance samples or AMD hardware qualification.

To rebuild, qualify and materialize collection workspaces on H100:

```bash
python3 tests/nbody_h100_profile.py \
  --device-uuid GPU-YOUR-H100-UUID --work-dir /tmp/nbody-h100-qualification
```

Use a new output directory. This compiles an isolated CUDA `sm_90` binary,
runs complete checks with the default warmer, and writes raw logs and
`profile.json`. Only a profile with `complete: true` is ready to use; a stopped
qualification leaves an incomplete checkpoint. It creates a `collection/timesteps-T` workspace for each
admitted timestep value, containing only its admitted body counts and a copy
of the exact qualified binary. Copy `gpubench` into each workspace before running
it: the harness prefers the executable's directory over the current directory
when locating a benchmark tree. Invoking a CLI inside another checkout can
select that checkout's full manifest and a different benchmark binary.

```bash
cd /tmp/nbody-h100-qualification/collection/timesteps-1
cp /absolute/path/to/gpubench ./gpubench
CUDA_VISIBLE_DEVICES=GPU-YOUR-H100-UUID ./gpubench run \
  --tier 2 --benchmark cuda_nbody --platform cuda --reps 7 --db results.sqlite
```

Repeat for each workspace listed in the generated `profile.json`. Preserve that
profile and the raw results together. The collection workspaces are for running
the copied binary; regenerate qualification when changing the binary, source,
compiler, GPU or workload. The general harness still has no built-in profile
filter. Whole-suite profile selection and provenance remain separate work.
CI checks this committed profile's current source hashes and candidate coverage.
Changes to benchmark inputs or any of the three shared headers require fresh
H100 qualification; updating hashes alone is not requalification. Run
`python3 tests/h100_profile_identity.py --benchmark cuda_nbody` from the repository
to check the source/configuration identity without a GPU.

Regression commands:

```bash
python3 tests/nbody_reference.py --compiler clang++
python3 tests/nbody_measured_result.py --platform cuda --work-dir /tmp/nbody-check
```

The host test checks analytic dynamics, every comparator component, tolerance
boundaries and production input/index guards under fatal sanitizers. The GPU
test corrupts actual measured position/velocity components at the first, middle
and last body, covers odd/even timestep buffers and bodies beyond the former
4096-body subset, and checks reference-resource failure. `--compile-only`
compiles its hooks without claiming GPU execution; CI uses this mode.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_nbody.hip`; tile size = 256 threads with `__shared__` memory
