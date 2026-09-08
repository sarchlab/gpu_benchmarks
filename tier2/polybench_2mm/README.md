# polybench_2mm — two dependent matrix products

The benchmark computes the PolyBench 2MM expression for square, single-precision
matrices:

```
tmp = alpha * A * B
D   = tmp * C + beta * D_initial
```

The first product completely overwrites the temporary. The second consumes that
temporary on the left, multiplies by C on the right, and adds the scaled initial
D. Alpha appears once, in the first stage; beta appears only in the final update.
Both stages run once per process. A 16×16 shared-memory tile implementation
handles dimensions that do not divide the tile width with zero padding.

## Upstream correspondence

The selected reference is [PolyBench-ACC's PolyBench/GPU 1.0 2MM source at
70ea4ca](https://github.com/cavazos-lab/PolyBench-ACC/blob/70ea4ca9047dc4ec9c3ca79b4c91b3b2cc1510e7/CUDA/linear-algebra/kernels/2mm/2mm.cu).
Its `mm2_cpu`, `mm2_kernel1` and `mm2_kernel2` define the same two-stage dependency
and matrix expression. This is an independent HIP/CUDA implementation: tiling,
thread mapping, and floating-point association differ from its scalar dot-product
kernels. Numerical qualification allows rounding differences.

The current workload configuration uses square dimensions controlled by N,
coefficients 1.5/1.2 by default, and four dense input matrices generated from
`rand()` seeded with 42, with values `(rand() % 100) / 10.0f`. These are explicit
suite inputs, not the pinned upstream's fixed initialization or published
configuration. That reference sets alpha=32412, beta=2123 and uses index-based
matrix formulas; the GPU regression test includes those formulas and coefficients
for square cases. The reference's separate NI/NJ/NK/NL controls are represented
here by one square dimension. Portable random input identity across C libraries
remains R-27, deferred by the owner.

Earlier revisions computed a different expression: an extra initial temporary
contribution, C on the left of the second product, and a second alpha factor.
Those results do not represent this workload and must be recollected. The repair
also removes the unused host temporary; random generation now fills four input
matrices, so historical seeded inputs should not be assumed identical.

## Build and run

```sh
cd tier2/polybench_2mm
make PLATFORM=cuda       # nvcc, default CUDA_ARCH=native
# make PLATFORM=rocm     # hipcc
./polybench_2mm
./polybench_2mm --size 2048
BENCH_PARAM_size=2048 ./polybench_2mm
```

The Makefile can auto-detect ROCm from `/opt/rocm`, otherwise CUDA. Both paths use
`../../common` headers.

| Environment variable | Argument | Default | Meaning |
|----------------------|----------|---------|---------|
| `BENCH_PARAM_size` | `--size` | 1024 | Square dimension N |
| `BENCH_PARAM_alpha` | `--alpha` | 1.5 | Scale first product |
| `BENCH_PARAM_beta` | `--beta` | 1.2 | Scale initial final-output matrix |
| `BENCH_PARAM_vector_add_warmup` | — | `true` | Warmer before the complete pass |

Environment variables take precedence. The existing 24 size points from 64 to
4096 and coefficient sweep values are unchanged. Invalid final extents,
non-finite coefficients, failed host allocations, and GPU runtime errors exit
nonzero. General override parsing/precision remains the R-16 qualification item.

Initial A/B/C/D uploads and allocation occur before measurement. The optional
warmer runs after uploads, directly before the first product; there is no warmer
between stages, preserving intermediate reuse. No initialization of the
intermediate is needed or timed. Repetition uses fresh processes
(`gpubench run --reps N`).

## Output and measurement

Stdout contains one record for each logical stage and a final summary. Example
with illustrative times:

```json
{"type":"kernel","name":"mm_kernel_1","iter":0,"time_ms":1.2,"host_time_ms":1.3,"params":{"size":1024,"alpha":1.5,"beta":1.2,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"mm_kernel_2","iter":0,"time_ms":1.3,"host_time_ms":1.4,"params":{"size":1024,"alpha":1.5,"beta":1.2,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":2.5}
```

Stage 1 is the overwrite specialization `mm_kernel<false>`; stage 2 is the
accumulating specialization `mm_kernel<true>`. Each event bracket covers one
kernel, and the summary sums those two GPU times. Host times additionally include
launch/synchronization costs. Device/setup and verification information go to
stderr. After timing, ordinary runs check every element of both measured
matrices against full independent double CPU products. The second reference
uses the CPU temporary, not the measured temporary. A mismatch returns exit
code 1 even though timing records have already been emitted. Both kernel
records identify `verification=full_cpu_v1`.

The reference preserves the qualified formula, increasing reduction-index
order per output, and `5e-4 * max(1, abs(reference))` tolerance. Non-finite
actual values, references and errors fail. Independent columns are blocked to
reuse matrix reads; no output or reduction term is skipped. Only the first ten
mismatch diagnostics are printed, while all mismatches are counted.

Verification adds two full device-to-host copies after timing and no GPU
kernel. The original host A and D allocations are reused for those copies only
after the CPU products have consumed all inputs. Two double reference matrices
need 256 MiB in total at N=4096, in addition to the existing 256 MiB of host
inputs; the two result downloads total 128 MiB. CPU work and copies are outside
GPU timing but count toward the process timeout. Qualify memory capacity and
the complete process runtime on each collection host.

The leading-order logical operation count is 4·N³ for two matrix products. Derive
GFLOPS downstream from that model and the summed GPU time. Scalar coefficients,
loop instructions and padded tile work make it different from a target's exact
instruction count. Neither this count nor per-stage event timing establishes an
end-to-end application rate.

## Reference qualification

```sh
make PLATFORM=cuda test  # or PLATFORM=rocm on a compatible GPU
./bench_output_test --rank-one 4096
```

The GPU test calls the same complete production pass as main. It compares both
intermediate and final matrices, with guarded buffers and a NaN-poisoned initial
temporary, against independent double-precision products on small, noncommuting
inputs. It covers tile boundaries, zero/negative coefficients and the pinned
upstream's square initialization. Large integer rank-one inputs have exact
closed-form references, allowing every output element to be checked through the
largest declared N without a cubic CPU reference. The tolerance is 5e-4 relative
with an absolute floor near zero; non-finite values always fail. These checks
establish the expression and data dependency, not performance.

CI compiles this GPU test; execution requires CUDA or ROCm hardware. Its CPU
oracle remains separate from the ordinary-run reference.

From the repository root:

```sh
c++ -O3 -std=c++14 -fsanitize=address,undefined tests/two_mm_reference_test.cpp -o /tmp/two-mm-reference
/tmp/two-mm-reference
python3 tests/two_mm_measured_result.py --platform cuda --work-dir /tmp/two-mm-measured-result
```

The host test checks 168 complete reference pairs against unblocked per-output
double products, including partial blocks and signed/zero coefficients, with
bit-identical values and unchanged inputs. It checks both tolerance branches
and non-finite rejection. The GPU regression uses a new output directory and
disposable instrumented source copies to run 158 ordinary-path cases: 53
controls and 105 actual measured-output faults, including both matrices,
first/prefix/middle/tail positions, finite and NaN/±Inf values, zero/negative
coefficients, and 12/24 simultaneous mismatches with capped diagnostics. CI
runs the host test and compiles the GPU regression; use `--compile-only` without
a GPU or `--platform rocm` on AMD.

NVIDIA validation does not establish AMD execution (R-19). Other-host process
budgets and trace-region/artifact qualification remain R-19/R-23/R-24.
