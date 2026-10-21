# tier2/polybench_3mm/hip — Three Chained Matrix Multiplications (PolyBench 3mm)

Benchmarks three chained matrix multiplications using GPU compute:

```
E = A * B
F = C * D
G = E * F
```

All matrices are square N×N single-precision floats (default N=256). The 3mm pass runs exactly once per execution and emits one time record per kernel; GFLOPS is derived downstream.

## Provenance & fidelity

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) — the `3mm` workload. Independent reimplementation, spot-verified to reproduce PolyBench's algorithm faithfully.

## Algorithm

- Three sequential GPU kernel launches per execution
- Each kernel: one thread per output element, simple dot-product inner loop (no tiling)
- Default size: N=256 (all matrices 256×256)
- GFLOPS = (2·N³ + 2·N³ + 2·N³) / time\_sec / 1e9 = 6·N³ / time\_sec / 1e9 (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `polybench_3mm.hip` | HIP source — three matrix-multiply kernels + self-contained utilities |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./polybench_3mm                       # defaults: 256×256
./polybench_3mm --size 512
BENCH_PARAM_size=512 ./polybench_3mm
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 256 | Matrix dimension N |
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the timed pass |

The 3mm pass runs exactly once per execution; repetition is the harness's job (`gpubench run --reps N`).

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per
kernel (`mm3_kernel1`, `mm3_kernel2`, `mm3_kernel3`) with that kernel's GPU
event time (`time_ms`) and host wall-clock time (`host_time_ms`), followed by
a `{"type":"summary","total_time_ms":...}` line whose value is the sum of the
three GPU event times. Derived metrics (GFLOPS) are computed downstream.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 256×256 (all)

k1 gpu = 0.712 ms, k2 gpu = 0.708 ms, k3 gpu = 0.705 ms
Total GPU time: 2.125 ms
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `polybench_3mm.hip`

## Matrix Dimensions

With `--size N`:

| Matrix | Rows | Cols |
|--------|------|------|
| A      | N    | N    |
| B      | N    | N    |
| E=A*B  | N    | N    |
| C      | N    | N    |
| D      | N    | N    |
| F=C*D  | N    | N    |
| G=E*F  | N    | N    |

## Measured-result verification

Every run checks every element of the measured E, F and G matrices. Complete
independent double CPU products compute E and F from the original A/B/C/D,
then G from those independent intermediates. GPU intermediates are never used
as reference input. All checking follows the three timed launches; three full
result downloads reuse the consumed host A buffer (all driver dimensions are
N). There is no separate GPU verification pass or bypass.

Actual, reference and error must be finite. E and F require
`abs(actual-reference) <= 5e-4 * max(1,abs(reference))`; G uses `2e-3` in place
of `5e-4` to cover propagation through both intermediate products. For the
declared nonnegative inputs and N <= 8192, these exceed the conservative
`gamma_N` and `gamma_(3*N)` bounds, respectively, where `u = 2^-24` and
`gamma_k = k*u/(1-k*u)`. The double-reference error is negligible relative to
these margins. Smaller errors are intentionally admitted; this does not
promise detection of arbitrary corruption or qualify every manual input.
All mismatches are counted, at most ten details go to stderr, and any mismatch
causes a nonzero exit. Each kernel record includes `verification="full_cpu_v1"`.

Each CPU product packs its right operand into contiguous column tiles and uses
at most sixteen workers on disjoint output row tiles. Packing preserves every
value and increasing reduction order. Products run sequentially, with one
worker below 512×512 output cells and otherwise a cap based on hardware
concurrency and row tiles. Thread creation failures, including allocation
failures, complete the missing partitions on the caller; prior workers are
joined before the result is returned and packed storage is released.
GPU kernels, inputs, geometry, initial uploads, warmer/event sequence and the
declared sweep are unchanged.

At N=8192, all 201,326,592 measured values are checked. Three double reference
matrices use 1.5 GiB; the largest temporary packed operand adds 512 MiB, giving
2 GiB of reference storage alongside the original 1 GiB host inputs. Packing
is released before the three 256 MiB result downloads, which reuse host A.
CPU work is outside GPU event time but inside the process timeout. Other-host
budgets/admission, AMD and full upstream qualification remain open.
