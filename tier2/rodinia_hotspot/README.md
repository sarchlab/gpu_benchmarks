# Rodinia HotSpot

Transient thermal simulation on an N×N grid using Rodinia's 16×16 shared-memory
pyramid. One process runs exactly `total_iterations` time steps, then checks every
actual final temperature against an independent FP64 CPU trajectory.

## Pinned problem and implementation

The selected reference is
[Rodinia 3.1 CUDA HotSpot](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/hotspot/hotspot.cu)
at `366b283456506ef2fe2c2b7dc5e83e941ffc5524`. The `calculate_temp` body and
physical-parameter expressions follow that source. University of Virginia
notices are retained in `UPSTREAM_LICENSE`; frozen fixtures are under
`tests/upstream/`.

The state and shared arrays are FP32. The original `2.0` literals promote much
of the stencil expression to FP64; this port preserves that mixed arithmetic.
Earlier local code changed those literals to `2.0f`. Results and timings from
that implementation should be recollected.

```
new = center + (dt/Cap) * (power
    + (south + north - 2.0*center) / Ry
    + (east  + west  - 2.0*center) / Rx
    + (80 - center) / Rz)
```

The kernel precomputes FP32 `dt/Cap` and reciprocal resistances, multiplies by
those reciprocals, and clamps all four missing neighbors to the center.
Each launch advances up to `pyramid_height` steps through shrinking shared-memory
interiors. It writes the nonoverlapping final region; earlier local steps also
update overlapping halos. The original host grid and border remain based on the
requested pyramid height, even for a shorter last launch. That tail can launch
extra inactive blocks, and its padding is included in admission and work counts.
The fixed step count uses integer progress and the current final buffer is
downloaded after all swaps.

Documented suite adaptations:

- Synthetic inputs retain `srand(42)` followed by interleaved
  `80 + float(rand()%200)/10` and `float(rand()%100)/500`. Upstream uses files.
  This libc-dependent recipe is not a portable cross-host dataset; portable
  input generation remains deferred.
- The upstream timestep (approximately 1.4583334e-7 seconds) is retained whenever
  the effective FP32 coefficients satisfy a conservative absolute-row-sum bound
  of at most one. Otherwise use `0.9*Cap/(2/Rx + 2/Ry + 1/Rz)`, recompute the
  coefficients and check the bound again. This is sufficient non-amplification,
  including clamped rows; it is not a necessary spectral stability test.
  In the declared profile, sizes through 1280 keep upstream dt; sizes 1536 and
  above use the limited dt. **Iteration counts remain fixed.** Actual timestep,
  limiting flag and simulated duration are recorded.
- The single entry warmer precedes the trajectory. All dependent launches are
  timed once and accumulated; no warmer is inserted between them. Host setup,
  input uploads, final download and CPU verification are outside timing.

## Build and controls

```
make PLATFORM=cuda                # or PLATFORM=rocm; plain make auto-detects
./rodinia_hotspot                  # N=1024, 360 steps, pyramid height 2
./rodinia_hotspot --grid_size 17 --total_iterations 3 --pyramid_height 7
```

The three flags have matching `BENCH_PARAM_` environment variables, which take
precedence. The first matching CLI flag is used; every supplied CLI value still
must parse as a positive 32-bit integer. Grid size is bounded by 46340 and pyramid
height by 1..7, then the actual full/tail padded index range and uint64 work counts
are checked. Device block, grid and shared-memory limits are checked before
allocation. An integer-admissible manual configuration is not a hardware or
runtime qualification. Host/device/reference allocation failure returns nonzero.

`BENCH_PARAM_vector_add_warmup=true` is the default; `false` disables the entry
warmer. It is a harness measurement control, not a workload sweep dimension.
The declared profile remains 20 sizes × three pyramid heights at 360 steps.

## Full verification and numerical bound

The CPU reference reads the original FP32 input and effective FP32 coefficients,
then evaluates all steps with FP64 arithmetic and explicit scalar clamping. It
uses one worker below N=512 and at most eight persistent workers otherwise,
capped by available hardware concurrency. Workers own disjoint output rows, and
every step completes before buffers swap. Partial thread-creation failure stops
and joins created workers and rejects the run. There is no sampling, smaller
verification problem or extra GPU reference pass in production.

Every actual and reference value must be finite. Their difference must be within
the following derived absolute bound. Let `g` bound the absolute coefficient row
sum; `h = 1 + sdc*(4*ix + 4*iy + iz)` bound the absolute expression magnitude;
`s = sdc*(max_power + 80*iz)` bound the source; `M` bound CPU state magnitude; and
`E` bound GPU-versus-CPU error. Here `sdc`, `ix`, `iy`, `iz` are the effective
FP32 divisions used by the kernel. For `X=M+E`, float unit roundoff `u=2^-24`,
`gamma2=2*u/(1-2*u)` and double `gamma24=24*2^-53/(1-24*2^-53)`:

```
Q = sdc * (2*u*(ix+iy)*X + gamma2*iz*(80+X))
D = gamma24 * (h*X + s + Q)
F = Q + D + u*(g*X + s + Q + D)
C = gamma24 * (h*M + s)
E_new = g*E + F + C + underflow_allowance
M_new = g*M + s + C + underflow_allowance
```

`Q` bounds the two FP32 neighbor sums and ambient subtraction/product; `D`
covers the intervening FP64 expression; the final term in `F` covers the FP32
store. FMA contraction reduces the number of rounding sites. The nonnegative
bound arithmetic rounds outward, includes underflow/flush-to-zero allowance and
rejects nonfinite bounds or possible FP32 overflow. This is a numerical agreement
bound, not a physical accuracy estimate or universal corruption detector.

The first ten mismatches are printed; all are counted, and any mismatch returns
status 1. Reference/storage failures also fail explicitly after timing rather
than claiming verification success. Peak host vectors require about 28*N² bytes;
GPU problem arrays require 12*N² bytes, excluding runtime and warmer storage.

## Output and validation

Stdout has one accumulated `calculate_temp` JSON record and one summary. Records
include controls, actual block/grid shape, cells, launches, launched threads,
logical cell updates and executed cell updates (including redundant halo work),
actual/upstream dt, limiting flag, simulated duration, verification bound and
worker count. Precision, algorithm, boundary, input and verification identifiers
make the selected problem explicit. GPU event and host launch intervals sum the
individual kernel launches; they exclude CPU verification.

See `../../tests/README.md` for commands. Host tests use pinned CUDA physical
parameters, an independently indexed expanded long-double operator, a scalar
mixed-precision model and exact comparisons across worker counts. GPU fixtures
compare complete results with the unchanged pinned kernel, using the approved
timestep for each grid. Additional instrumentation counts actual GPU updates
and corrupts actual final output in isolated test binaries. H100 qualification
is separate from AMD hardware, trace/replay and other-host resource admission.
