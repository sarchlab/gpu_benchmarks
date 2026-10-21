# NAS MG (native CUDA)

This benchmark runs the FP64 periodic multigrid algorithm from
[GMAP/NPB-GPU, HIP/MG/mg.cpp](https://github.com/GMAP/NPB-GPU/blob/3f12d84920ee315ab00ef283717c1e74b68f4d00/HIP/MG/mg.cpp).
The pinned revision is `3f12d84920ee315ab00ef283717c1e74b68f4d00`.
The numerical GPU kernel bodies are retained in `mg_kernels.h`; suite wrappers
provide admission, allocation, timing, and complete-result verification. The
original MIT license and NAS authorship notices accompany the adapted code.

## Class domain

| `npb_class` | Interior grid | V-cycles | Levels (coarsest interior 2³) | Published final L2 |
|---|---:|---:|---:|---:|
| S | 32³ | 4 | 5 | 0.5307707005734e-04 |
| W | 128³ | 4 | 7 | 0.6467329375339e-05 |
| A (default) | 256³ | 4 | 8 | 0.2433365309069e-05 |
| B | 256³ | 20 | 8 | 0.1800564401355e-05 |
| C | 512³ | 20 | 9 | 0.5706732285740e-06 |

The five-value scaling axis is a standard NAS class-domain exception to sweep
density guidance. A and B have the same grid but distinct cycle counts and
smoother coefficients. Each level includes one periodic ghost cell on each side.
The operator is `a=(-8/3,0,1/6,1/12)`. The smoother is
`c=(-3/8,1/32,-1/64,0)` for S/W/A and `c=(-3/17,1/33,-1/61,0)` for B/C.

`BENCH_PARAM_npb_class` selects the class; `--class A` takes precedence.
The old `grid_size`, `num_vcycles`, `num_smooth_steps` environment controls and
`--size` CLI are rejected. They described a different, custom solver and cannot
select an unambiguous standard class. No input is rounded or silently clamped.

`block_size` defaults to 256 and admits powers of two from 1 through 1024,
subject to device limits. The residual and smoother use `min(padded_width,block_size)`
threads per row; the norm, boundary exchanges and coarse zeroing use `block_size`.
Restriction retains `coarse_padded_width-1` threads per row and interpolation
retains `coarse_padded_width`. These fixed row geometries are requirements of the
pinned shared-memory kernels. Every record reports its actual launch count,
aggregate launched threads and minimum/maximum block width across levels.

## Input and execution

NAS `zran3` selects ten positive and ten negative spikes from its complete
x-fastest random grid. The generator uses seed 314159265, multiplier 1220703125
and the exact low 46 product bits. Streaming the extrema avoids retaining a
random temporary grid; strict comparisons, tie ordering, draw order and periodic
ghosts match the pin. Tests compare complete inputs against the original
floating-point `randlc`/`vranlc`/`zran3` implementation.

One process performs one complete NAS execution:

1. Prepare input and zero initial solution/storage outside measurement, then run
   the optional entry vector-add warmer once.
2. Compute the finest residual and its initial L2/Linf norm.
3. For each prescribed V-cycle, restrict residuals to the coarsest level, zero
   and smooth the coarse solution, then interpolate corrections upward. Every
   intermediate solution is explicitly zeroed before interpolation. The finest
   solution retains the preceding cycle's approximation. Residual and smoothing
   steps and all three periodic boundary exchanges follow the pinned stage order.
4. Compute the outer finest residual after every cycle and the final L2/Linf norm.

Every defining GPU launch, including coarse zeroing and periodic exchanges,
is timed. The upstream preliminary warmup solve is omitted under the suite's
one-execution policy. The initial and final norm reductions retain GPU row
partials and serial host finishing; partial transfers/host finishing are outside
summed kernel time. There is no reference calculation between measured stages.
Records accumulate one entry per kernel kind, `iter=0`, followed by their summed
GPU-time summary. Launch totals are 262/374/430/2126/2406 for S/W/A/B/C.

The warmer intends to prepare entry cache and clock state; it does not prove
residency or a particular clock rate. The full hierarchy evolves naturally
within the algorithm.

## Verification and qualification

After measurement, download **all** final solution and residual values at every
level, all periodic faces/edges/corners, and the actual finest input. An independent
CPU execution of the pinned serial recurrence regenerates its own input and
checks those complete arrays and both actual L2/Linf norm pairs. There is no
smaller verification problem, sampling, residual-ratio acceptance, or fallback.
The reference uses up to eight independent plane workers, preserving each cell's
arithmetic order and every phase boundary. Norm accumulation remains serial.
Allocation or worker-creation failure makes the process fail verification.

Input spikes and ghosts must match exactly. All actual/reference values must be
finite; computed state/norm pairs use
`abs(actual-reference) <= 1e-11 * (1+abs(reference))` to allow FP64 contraction and
reduction-order differences. This is a mixed absolute/relative bound: near-zero
values receive an absolute accuracy check. In addition, the actual final L2 must
match its published class value within the upstream relative `1e-8` threshold.
Neither check substitutes for the other. Verification failures return nonzero;
the runner retains raw output but cannot treat it as a successful measurement.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. The H100 collection domain consists only of the standard cases passing complete
verification within the runner's existing 60-second process limit. This repair
qualifies that domain with `tests/mg_measured_result.py`; qualification timings
are functional evidence, not collected publication performance results.
Trace/capture/replay work remains deferred.

## Checks

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. - `python3 tests/mg_host.py --compiler g++`: frozen Class S full trajectory,
  46-bit RNG, corruption rejection, launch/class admission and worker cleanup,
  under fatal host sanitizers. `--all-inputs` compares S--C complete inputs;
  `--thread-sanitizer` checks the reference workers on a supported host.
- `python3 tests/mg_measured_result.py --platform cuda --work-dir /tmp/mg-check`:
  all five production classes, launch controls, measured device-array faults at
  every S hierarchy level/ghost region, input/norm faults, largest-class tail
  corruption and resource failures. Each process has a 60-second limit.
- `make -C tier2/npb_mg/cuda test`: actual FP64 coarse-zero output and
  guard checks, including the largest padded coarse level and partial blocks.
- `tests/finite_verification.py` retains actual/reference nonfinite fault hooks.
  `tests/algorithmic_zero_order.py` describes the updated launch order for later
  optional profiler validation; no trace qualification is claimed here.
