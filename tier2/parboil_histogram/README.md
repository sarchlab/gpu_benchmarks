# tier2/parboil_histogram — Large 2D Saturating Histogram (Parboil histo)

Faithful port of the Parboil **histo** benchmark: the four device kernels are the
original (`parboil/benchmarks/histo/src/cuda/`), the host driver reproduces the
original `main.cu` geometry. Only the host driver is restructured for the
single-run timing harness and JSON output.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `histo` workload. Independent reimplementation, spot-verified to reproduce Parboil's algorithm faithfully (includes 8-bit saturation + overflow buffer).

## Algorithm (4-kernel privatized pipeline — the faithful part)

Builds a `histo_width × histo_height` 2D histogram with **8-bit saturating** bins:

1. `histo_prescan_kernel` — samples the input, estimates avg/stddev, derives the
   active histogram-block range `[sm_range_min, sm_range_max]`.
2. `histo_intermediates_kernel` — decomposes each input value into its
   `(block, hi, lo, offset)` bin coordinate (`uchar4` mappings).
3. `histo_main_kernel` — each threadblock owns one 24 KB histogram block and keeps
   a **privatized sub-histogram in shared memory** with 8-bit saturating atomics
   (overflow tracked in a global buffer).
4. `histo_final_kernel` — **saturating merge** of the `BLOCK_X` sub-histograms and
   the global/overflow buffers into the final 8-bit histogram.

The single run executes all four kernels once; each kernel is emitted as its
own record. Mutable state (`ranges`, sub-histogram, global/overflow buffers)
is initialised untimed before the run.

## Default problem size

Histogram shape **256 × 8192 = 2,097,152 bins** (≈ 85 24 KB blocks), matching the
original's 2D histogram shape. Input image **256 × 4096 ≈ 1.05 M** elements.
Input values are drawn from a narrow near-normal band (mimicking the spatially
coherent real image data), keeping the prescan-derived active range well inside
the valid block count. `img_height` should be a multiple of 16 (`UNROLL`), and
`img_width` is kept equal to `histo_width` (as in the reference dataset).

## Measurement

The 4-kernel pipeline runs **exactly once** per process execution (see the root
README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before the timed run. Repetition =
`gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_img_width` | Input width (= histo_width) | 256 |
| `BENCH_PARAM_img_height` | Input height (multiple of 16) | 4096 |
| `BENCH_PARAM_histo_width` | Histogram width | 256 |
| `BENCH_PARAM_histo_height` | Histogram height | 8192 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Build / Run

```bash
make               # auto-detect platform (cuda / rocm)
./parboil_histogram
BENCH_PARAM_img_height=8192 ./parboil_histogram
```

## Output

**stdout** — one `kernel` record (`iter:0`) for **each** of the four kernels
carrying that kernel's GPU event time within the single run, then a `summary`
whose `total_time_ms` is the sum of all recorded GPU event times.
**stderr** — device info, per-kernel timing, PASS/FAIL. Verification:
`final_histo[value] == min(count(value), 255)` over all bins; on FAIL the
process exits non-zero.
