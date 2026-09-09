# Rodinia Needleman–Wunsch

A complete GPU dynamic-programming alignment of two sequences using BLOSUM62
substitution scores and a linear gap penalty. Every cell of the actual measured
DP matrix is checked exactly against a separate CPU recurrence after timing.

## Pinned upstream and fidelity

The reference is Rodinia 3.1 at
[`366b283456506ef2fe2c2b7dc5e83e941ffc5524`](https://github.com/HPC-FAIR/rodinia_3.1/tree/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/nw):
[`needle.cu`](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/nw/needle.cu)
defines the input recipe, 24×24 BLOSUM62 table and gap borders;
[`needle_kernel.cu`](https://github.com/HPC-FAIR/rodinia_3.1/blob/366b283456506ef2fe2c2b7dc5e83e941ffc5524/cuda/nw/needle_kernel.cu)
fills tiles in two dependency-ordered wavefront phases.
The copied table's upstream copyright, conditions and disclaimer are retained
in the HIP source; reproduce that notice with binary distributions.

The driver calls `srand(7)`, draws all row-sequence symbols using `rand()%10+1`,
then draws all column-sequence symbols using the same recipe. It precomputes
`reference[i,j] = BLOSUM62[row[i]][column[j]]` on the CPU, initializes the DP
interior to zero, and sets borders `D[i,0]=-i*penalty`, `D[0,j]=-j*penalty`.
The GPU evaluates every interior cell exactly once:

```text
D[i,j] = max(D[i-1,j-1] + reference[i,j],
             D[i-1,j] - penalty,
             D[i,j-1] - penalty)
```

This is an independent HIP/CUDA implementation. Dynamic shared memory and the
within-tile thread mapping differ from upstream; the substitution problem,
recurrence, complete reference-tile loads, gap boundaries and two wavefront
phases are preserved. Upstream's optional CPU traceback is disabled in its
normal build and is not part of this measured workload.

The previous local benchmark aligned a single seed-42 sequence with itself
using +1/−1 equality scores and had no measured-output check. Those results
represent a different workload and need recollection. The recorded algorithm
identity is now `rodinia31_blosum62_v1`.

## Parameters and effective work

Build with `make PLATFORM=cuda` or `make PLATFORM=rocm`. Both use the same HIP
source and shared timing headers. Example invocations:

```bash
./rodinia_nw
./rodinia_nw --sequence_length 4096 --block_size 32 --penalty 10
BENCH_PARAM_sequence_length=33 ./rodinia_nw
```

| CLI option / `BENCH_PARAM_` suffix | Default | Meaning |
|---|---|---|
| `sequence_length` | 2048 | Requested sequence length |
| `block_size` | 16 | Tile edge and threads per block |
| `penalty` | 2 | Positive linear gap penalty |
| `vector_add_warmup` (environment only) | true | Standard entry warmer |

Environment values override CLI values. The existing positive CLI fallback is
preserved. The 80 declared size/penalty configurations and block-size default
remain unchanged. Manual block sizes, including 32, are unswept controls.

As before, a manual length rounds **up** to a multiple of `block_size`. With
requested 33 and block size 16, the problem contains two sequences of length
48 and a 49×49 DP matrix. This is an explicit extension to the pinned driver's
aligned-size interface; comparisons use the effective length. Records include
requested `sequence_length`, `effective_sequence_length`, `matrix_cells`,
`total_cell_updates`, `block_size`, `penalty`, `input_seed`, algorithm identity
and verification policy. Each kernel row also records its own `kernel_launches`
and `cell_updates` so per-kernel work is not confused with the whole fill.

Nonpositive values, unsafe padded matrix indices or possible signed-score
overflow are refused before device access. Admission requires
`(effective_length+1)^2 <= INT_MAX` and
`(effective_length+1)*max(penalty,11) <= INT_MAX`; the latter conservatively
bounds even losing recurrence candidates. Device thread, grid and shared-memory
limits are checked before allocation. Allocation/reference failures return an
error. These checks do not establish admission on every host or GPU.

## Measurement and verification

Input generation, allocation and initial uploads occur before the one entry
warmer. The complete upper-left and lower-right wavefront phases then run once,
with one event bracket per launch and no reconditioning between dependent
launches. Each kernel's accumulated event time is emitted as one JSON record;
the summary is their sum. A one-tile problem launches only `nw_kernel1` and
emits no fictitious second-kernel record. Repetition is the harness's job.

After timing, the driver downloads the complete measured DP matrix. The CPU
verifier uses the retained sequences and BLOSUM62 table to compute an independent
row-major recurrence with two rows of scratch space. It does not use measured
DP values or the uploaded score matrix as recurrence inputs. It compares all
interior and border cells with exact integer equality, counts every mismatch,
prints at most ten numeric diagnostics, and returns failure on any disagreement
or unavailable reference. The policy is `full_dp_cpu_v1`; no separate GPU check,
sampling or tolerance is used.

GPU storage is two full integer matrices, plus the shared entry warmer. Peak
host storage is approximately `8*(effective_length+1)^2 + 16*(effective_length+1)`
bytes, including both matrices, both sequences and the CPU rows. Full CPU
verification is quadratic and counts toward the harness's 60-second process
budget. The general resource/profile work still governs collection admission.

For tile edge B and T=`effective_length/B`, phase 1 updates
`T*(T+1)/2 * B^2` cells and phase 2 updates `T*(T-1)/2 * B^2` cells. Their sum
is `effective_length^2`. GCUPS may be derived from those logical counts and
raw times. These counts do not establish physical memory traffic or cache hits.

## Validation

```bash
python3 tests/nw_reference.py --compiler clang++
CUDA_ARCH=sm_90 python3 tests/nw_measured_result.py \
  --platform cuda --sweep --work-dir /tmp/nw-check
```

Use a fresh output directory. Host tests check the pinned table, input recipe,
analytic alignments, full reference, corruption rejection and actual input
bounds under fatal sanitizers. GPU tests run the production binary, corrupt
actual measured cells including borders, perturb uploaded scores, and inject
reference-resource failure. `--sweep` also runs every declared configuration;
CI compiles hooks with `--compile-only` without claiming GPU execution.

On the qualified H100 PCIe host (CUDA 12.9.86, GCC 11.4), all 80 declared
configurations pass full verification with the default warmer. The 122-case
production/control/fault run takes at most 4.723 seconds per process; the
largest matrix contains 604,028,929 checked cells. Sixteen complete GPU matrices,
including the largest case and rounded/manual controls, match the pinned
Rodinia GPU implementation byte for byte. Four GPU sanitizer runs are clean;
six native CLI/SQLite cases preserve descriptors and success/failure status.
These are correctness and process-budget qualification runs, not publication
performance samples or a promise about other hosts. The reference uses one
CPU worker; the admitted profile must also fit the target host's memory budget.

The C-library RNG recipe is pinned, but identical input bytes across different
C libraries are not promised. Portable input generation and AMD hardware
qualification remain deferred. Trace/replay qualification is separate.
