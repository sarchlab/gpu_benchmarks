# NAS Conjugate Gradient (CG) (native CUDA)

Runs the NAS CG inverse-power benchmark in FP64 for standard classes S, W, A,
B and C. Each process performs one complete benchmark problem and then verifies
its measured results. The harness supplies repetition across fresh processes.

## Provenance and algorithm

The implementation follows [GMAP/NPB-GPU, commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00`, CUDA/CG/cg.cu](https://github.com/GMAP/NPB-GPU/blob/3f12d84920ee315ab00ef283717c1e74b68f4d00/CUDA/CG/cg.cu)
and its `CUDA/common/c_randdp.cpp` generator. `nas_cg.h` retains the upstream
MIT license and NAS attribution.

The input uses the prescribed NAS sparse outer-product matrix construction,
46-bit `randlc` stream (seed 314159265, multiplier 1220703125, one discarded
initial draw), `RCOND=0.1`, and the class-specific diagonal shift. Row incidence
lists replace insertion moves during host matrix assembly while preserving
every contribution's addition order and sorted CSR column order. Regression
checks compare every matrix value and CSR index with the pinned `makea` code.

Starting with `x=1`, each inverse-power iteration performs 25 inner CG iterations,
an explicit residual SpMV and norm, two normalization dot products, the zeta
estimate, and vector normalization. There is no convergence early exit or
clamping of denominators. Non-finite arithmetic fails the process.

The portable GPU mapping assigns one thread to each CSR row. The pinned CUDA
mapping uses a block per row. Independent z/r updates share one launch here;
their mathematical updates and dependencies remain unchanged. Block reductions
produce partials that the host finishes in order, as in the selected upstream
hybrid implementation. Host scalar calculations remain outside GPU event time.

The upstream preliminary CPU solve is omitted under the repository's single
execution policy. Initialization and input upload occur before one entry warmer.
The warmer is intended to establish entry clock/cache conditions; it is not
proof of a particular clock or cache state. The full dependent algorithm then
runs without additional warmers or verification between stages.

## Standard problem classes

| Class | `matrix_size` | NONZER | Outer iterations | SHIFT | Published zeta |
|---|---:|---:|---:|---:|---:|
| S | 1400 | 7 | 15 | 10 | 8.5971775078648 |
| W | 7000 | 8 | 15 | 12 | 10.362595087124 |
| A (default) | 14000 | 11 | 15 | 20 | 17.130235054029 |
| B | 75000 | 13 | 75 | 60 | 22.712745482631 |
| C | 150000 | 15 | 75 | 110 | 28.973605592845 |

`NONZER` controls the upstream sparse generating vectors; it is not the final
CSR row length. The emitted `matrix_nonzeros` reports the actual CSR count.
Five scaling values are an intentional NAS class-domain exception to the usual
dense size sweep. The former custom sparsity/iteration cross-product does not
represent NAS CG and has been replaced with these five complete problems.

| Environment variable | Default | Accepted values |
|---|---:|---|
| `BENCH_PARAM_matrix_size` | 14000 | The five class sizes above |
| `BENCH_PARAM_block_size` | 256 | Power of two from 1 through 1024, within device limits |
| `BENCH_PARAM_vector_add_warmup` | true | `true` or `false` |

The manifest collects all five classes at block size 256. `--size N` is retained
and takes precedence over `BENCH_PARAM_matrix_size`. Integers must parse fully
and fit a signed int. Unknown/missing arguments, unsupported classes and obsolete
`BENCH_PARAM_nonzeros_per_row` or `BENCH_PARAM_max_cg_iterations` variables are
rejected before device access. Malformed environment integers are rejected even
when a CLI size was supplied.

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./npb_cg                          # Class A
./npb_cg --size 150000            # Class C
BENCH_PARAM_matrix_size=1400 ./npb_cg
```

## Timing and work records

Stdout contains six JSON kernel records and one summary. Each kernel record
accumulates all launches of that kernel in the complete inverse-power problem.
Per outer iteration, the counts are:

| Kernel | Launches |
|---|---:|
| `cg_init_kernel` | 1 |
| `spmv_kernel` | 26 |
| `dot_product_kernel` | 54 |
| `axpy_kernel` | 25 |
| `direction_kernel` | 25 |
| `normalize_kernel` | 1 |

`kernel_launches` reports the actual accumulated count, and
`spmv_nonzero_products` is actual CSR nonzeros times SpMV launches (zero on other
records). Descriptors include the full NAS problem, input recipe, FP64 precision,
actual grid, reference worker count, `algorithm=npb_cg_inverse_power_v1`, and
`verification=full_cpu_nas_zeta_v1`. These version fields distinguish this restored
NAS workload from historical synthetic-CG records.

`time_ms` sums GPU event times. `host_time_ms` sums the corresponding timed launch
intervals. `total_time_ms` sums the six GPU records; it excludes host reduction
finishing, scalar calculations, transfers, setup and verification. It is not an
end-to-end solution time. There are 1980 GPU launches for S/W/A and 9900 for B/C.

## Verification and resources

After timing, the process downloads all five actual final vectors (`x,z,p,q,r`)
and compares every entry, plus every retained outer zeta and residual norm, with
a complete independent CPU recurrence. No second GPU execution supplies the
checked result. Both values must be finite and satisfy
`abs(actual-reference) <= 1e-10 * (1 + abs(reference))`. The final zeta must also
meet the upstream published value within relative `1e-10`. Failure returns
nonzero; diagnostics show up to ten numeric mismatches and count all errors.

The CPU reference preserves serial row accumulation and scalar reduction order.
Classes S/W/A use one CPU worker. B/C use up to eight persistent row workers,
bounded by hardware concurrency. Worker creation or allocation failure fails
verification and cannot yield an accepted result. The benchmark requires host
memory for complete matrix generation and the reference as well as device
storage for CSR, five vectors and reduction partials. Matrix generation reserves
up to `12*N*(NONZER+1)^2` bytes for CSR values/columns plus temporary input and row
assembly storage; the Class C CSR reservation alone is 460.8 MB. The optional
entry warmer adds its own device working set.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. H100 qualification covers all five declared configurations with the full
reference under the existing 60-second per-process limit. Host core availability
and memory contention can affect this limit. Trace/capture/replay work remains
deferred.