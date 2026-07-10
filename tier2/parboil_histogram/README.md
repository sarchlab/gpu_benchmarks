# tier2/parboil_histogram — Large 2D Saturating Histogram (Parboil histo)

Faithful port of the Parboil **histo** benchmark: the four device kernels are the
original (`parboil/benchmarks/histo/src/cuda/`), the host driver reproduces the
original `main.cu` geometry. Only the host driver is restructured for the
per-iteration timing harness and JSON output.

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

One recorded **pass** runs all four kernels; each kernel is emitted as its own
per-iteration record. Per-pass state (`ranges`, sub-histogram, global/overflow
buffers) is re-initialised untimed at the start of each pass.

## Default problem size

Histogram shape **256 × 8192 = 2,097,152 bins** (≈ 85 24 KB blocks), matching the
original's 2D histogram shape. Input image **256 × 4096 ≈ 1.05 M** elements.
Input values are drawn from a narrow near-normal band (mimicking the spatially
coherent real image data), keeping the prescan-derived active range well inside
the valid block count. `img_height` should be a multiple of 16 (`UNROLL`), and
`img_width` is kept equal to `histo_width` (as in the reference dataset).

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_img_width` | Input width (= histo_width) | 256 |
| `BENCH_PARAM_img_height` | Input height (multiple of 16) | 4096 |
| `BENCH_PARAM_histo_width` | Histogram width | 256 |
| `BENCH_PARAM_histo_height` | Histogram height | 8192 |
| `BENCH_PARAM_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed passes, each recorded | 7 |

## Build / Run

```bash
make               # auto-detect platform (cuda / rocm / metal)
./parboil_histogram
BENCH_PARAM_img_height=8192 ./parboil_histogram
```

## Output

**stdout** — per timed pass, one `kernel` record for **each** of the four kernels
carrying that kernel's GPU event time, then a `summary` whose `total_time_ms` is
the sum of all recorded GPU event times. **stderr** — device info, per-pass
timing, PASS/FAIL. Verification: `final_histo[value] == min(count(value), 255)`
over all bins.

> **Metal note:** `parboil_histogram.metal` / `parboil_histogram_metal.mm` still
> contain the earlier naive 256-bin atomic implementation and were not updated by
> this faithful port (HIP/CUDA only).
