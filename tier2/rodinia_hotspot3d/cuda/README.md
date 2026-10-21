# Rodinia HotSpot3D (native CUDA)

An FP32 thermal stencil on an N×N×N grid. One timed pass performs exactly
`num_iterations` updates, then checks every temperature in the actual final
measured buffer against an independent FP64 CPU reference. Harness repetitions
start fresh processes; verification and input generation are outside timing.

## Source and fidelity

The physical model and update expression are pinned to Rodinia 3.1,
[HPC-FAIR/rodinia_3.1 at 366b283456506ef2fe2c2b7dc5e83e941ffc5524](https://github.com/HPC-FAIR/rodinia_3.1/tree/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/hotspot3D):
`3D.cu` (parameters and CPU stencil) and `opt1.cu` (GPU expression).
The upstream University of Virginia license is in `UPSTREAM_LICENSE`.

The recurrence, physical constants, coefficient construction and GPU summation
order follow that source. All six faces clamp the missing neighbor to the center.
The ambient source applies to every cell:

```
new = cc*center + cw*west + ce*east + cs*south + cn*north
    + cb*bottom + ct*top + (dt/Cap)*power + ct*ambient
ce = cw = dt/(Cap*Rx); cn = cs = dt/(Cap*Ry); ct = cb = dt/(Cap*Rz)
cc = 1 - (2*ce + 2*cn + 3*ct)
```

Documented adaptations:

- One thread updates one cell in B×B×B blocks. Upstream marches through z in
  registers with 64×4 blocks. This changes memory traffic and execution mapping;
  results are not interchangeable with upstream performance measurements.
- Retain this repository's synthetic input: `srand(42)`, then interleaved
  `ambient + float(rand()%200)/10` and `float(rand()%100)/500`. Upstream reads
  temperature and power files. This libc-dependent recipe is not a portable
  cross-platform dataset; portable input generation remains deferred.
- Ceiling grids and guards support partial blocks. N=1 is a defined extension.
  Ambient temperature is configurable, while the upstream GPU fixes it at 80.
- Download the current source after the final ping-pong swap. Upstream's wrapper
  downloads the other pointer; copying that wrapper would check a stale step.
- Use the upstream timestep when its FP32 coefficients satisfy the sufficient
  non-amplification bound `sum(abs(cc,cw,ce,cs,cn,cb,ct)) <= 1`. Otherwise use
  `dt = 0.9*Cap/(2/Rx + 2/Ry + 3/Rz)` and check the bound again. This conservative
  condition includes clamped boundary rows. It is not a necessary spectral
  stability test. **All 66 declared size/step pairs retain the upstream timestep**
  (approximately 1.4583334e-7 seconds). Only larger manually selected grids need
  limiting. Iteration counts remain fixed, so simulated duration can change.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. Frozen upstream CPU and GPU bodies are kept under `tests/upstream/`. The GPU
fixture only adds an x/y bounds guard to run a ceiling grid. Its z-marching body
requires N>=2. The CPU fixture returns the actual final pointer after swapping.
Neither fixture is used as the production CPU verifier. The GPU comparison
checks both complete outputs against the FP64 trajectory and its derived bound.
Different mappings can contract different products into FMAs, so equal source
expressions do not imply bitwise-equal FP32 results.

## Build and controls

```
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./rodinia_hotspot3d         # N=64, B=8, 10 updates, ambient=80
./rodinia_hotspot3d --grid_size 17 --num_iterations 3 --amb_temp 79.9
```

`--grid_size`, `--block_size`, `--num_iterations` and `--amb_temp` have
`BENCH_PARAM_` environment equivalents. Environment values take precedence;
the first matching CLI flag is used. Every supplied CLI value must still be
syntactically valid. Integers require full parsing, positive values and no
overflow. N is restricted to 1..1290 for 32-bit cell indices, B to 1..10, then
actual device geometry limits are checked. Ambient must be finite and nonzero
values must remain finite and nonzero after FP32 conversion. Allocation or
reference failure returns nonzero.

`BENCH_PARAM_vector_add_warmup=true` (default) runs the shared warmer once before
the trajectory. Steps then depend on one another without additional warmers.
Set it to `false` to disable that entry warmer. GPU allocations use 12*N³ bytes;
peak host vectors use approximately 28*N³ bytes, excluding runtime overhead.
Integer-admissible manual dimensions are not a hardware or runtime qualification.

## Verification

The serial CPU reference starts from the exact FP32 input and coefficients,
then evaluates the complete trajectory in FP64 with explicit scalar indexing
and six-face clamping. Every final GPU element must be finite and within a
derived absolute error bound of the finite CPU result. There is no sampling,
GPU reference rerun, or permissive relative tolerance. The first ten mismatches
are printed; all mismatches are counted and cause exit status 1.

Let `g` be the absolute coefficient row sum, `s` the bound on the power and
ambient terms, `M` a bound on CPU magnitude and `E` on GPU-versus-CPU error.
Each step has nine products and eight additions. With `gamma_p = 18*u_p/(1-18*u_p)`
for FP32 and FP64, the implementation propagates:

```
next = g*M + s
E_new = g*E + gamma_32*(next + g*E) + gamma_64*next + underflow_allowance
M_new = next + gamma_64*next + underflow_allowance
```

The nonnegative bound arithmetic rounds outward. An allowance of 18 times the
smallest normal FP32 value covers underflow/flush-to-zero; FMA uses fewer
rounding operations than the bound permits. Nonfinite bounds or possible FP32
overflow reject the run. This bound validates numerical agreement for this
recurrence; it is not a physical accuracy estimate. More iterations can widen
it. The declared maximum remains 50 updates. Host tests compare the reference
with a separately indexed long-double oracle and pinned FP32 CPU trajectories.

## Output and qualification

Stdout contains one accumulated `hotspot3d_kernel` JSON record and one summary.
GPU event and host launch time sum the individual step intervals. Metadata
includes controls, actual launch geometry, cells, launched threads per step,
launch count, total cell updates, actual and upstream timestep, whether the
timestep was limited, simulated duration and verification bound. `amb_temp`
retains the requested numeric value for harness matching; `effective_amb_temp`
records the FP32 value actually used. Algorithm/input/precision/boundary and
verification identifiers distinguish this workload from other variants.
Stderr reports the complete verified count, mismatch count and verdict.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. See `../../../tests/README.md` for host, measured-output fault and original-GPU
comparison commands. H100 qualification covers the declared profile and selected
manual controls. Trace/replay qualification remains separate.