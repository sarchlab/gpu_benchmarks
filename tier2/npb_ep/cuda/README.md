# tier2/npb_ep/cuda — Embarrassingly Parallel (NAS EP) (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

Faithful port of the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) **EP** kernel, taken from [NPB-GPU (GMAP)](https://github.com/GMAP/NPB-GPU). The device kernel (`gpu_kernel`) and its random-number generators (`randlc_device`, `vranlc_device`) are the **original upstream source**; only the host driver is restructured for the single-run timing harness and JSON output.

EP generates 2^(M+1) pseudorandom numbers, forms Gaussian deviate pairs, and tallies them into 10 concentric square annuli. It is purely compute-bound with negligible memory traffic — the canonical "embarrassingly parallel" workload.

## Provenance & fidelity

Faithful port of the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) `EP` kernel: `randlc`/`vranlc` + skip-ahead reproduced verbatim, and the reference sums match published NPB digit-for-digit. Only the host driver is restructured for the single-run timing harness and JSON output.

## Algorithm (original NAS EP)

- Generate 2^(M+1) uniform numbers with the NAS 46-bit linear congruential generator `randlc`/`vranlc` (R23/R46 split, `A = 5^13 = 1,220,703,125`, seed `S = 271828183`).
- Each of the 2^(M-16) threads derives its own starting seed via the `randlc` skip-ahead loop, then draws `NK = 2^16` pairs of numbers in batches of `RECOMPUTATION = 128` pairs.
- **Marsaglia polar (acceptance-rejection)** method: for a pair `(x1, x2)` in [-1,1]², accept iff `t1 = x1²+x2² <= 1`, then `t = sqrt(-2 ln t1 / t1)`, `t3 = x1·t`, `t4 = x2·t`.
- Annulus bin `l = max(|t3|, |t4|)` (integer truncation); increment `q[l]` and accumulate `Sx += t3`, `Sy += t4`.
- Each block reduces its threads' partials via `atomicAdd` into per-block slots; the host sums the per-block partials after the timed loop.

This replaces the previous (incorrect) ANSI-C LCG + Box-Muller reimplementation with the real NAS algorithm.

## Class / problem size

The class exponent **M** sets the size. Default **M=28 (Class A)** — 2^29 numbers over 2^12 threads, which runs in well under a second on a modern GPU while giving a substantial per-pass kernel time.

| M  | Class | Numbers (2^(M+1)) |
|----|-------|-------------------|
| 24 | S     | 2^25              |
| 25 | W     | 2^26              |
| 28 | A     | 2^29 (default)    |
| 30 | B     | 2^31              |
| 32 | C     | 2^33              |

## Files

| File | Description |
|------|-------------|
| `npb_ep.cu` | CUDA source — original NPB-GPU EP kernel + timing harness |
| `Makefile` | nvcc build (`CUDA_ARCH`, default `native`) |
| `params.json` | Parameter specification |

## Build

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

## Run

```bash
./npb_ep                 # defaults: M=28 (Class A)
./npb_ep --m 24          # Class S
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_m` | 28 | Class exponent M (24 S, 25 W, 28 A, 30 B, 32 C) |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed launch; set `false` to time as-is |

Resolved `M` must be greater than 16, and `2^(M-16)` must fit the positive
`int` thread-count argument (M=17 through 46 with 32-bit `int`). The block size
must be positive, and the last per-block histogram index,
`ceil(2^(M-16) / block_size) * 10 - 1`, must also fit `int`. Invalid inputs fail
before allocation or device access. These are representation checks; they do
not establish device block-size support, available memory, or a process budget
for large direct overrides. All 15 declared configurations remain supported by
these checks. Nonstandard M values without reference sums retain the existing
verification-skipped behavior and are not qualified by passing the input checks.

## Timing model

EP is embarrassingly parallel: the benchmark times **one `gpu_kernel` launch per execution** (the per-block partial reduction is inside the kernel); repetition across fresh processes is the harness's job (`gpubench run --reps N`). The RNG seeds are derived from fixed constants inside the kernel, so every launch performs identical work — no input reset is needed. The per-block output accumulators are written via `atomicAdd`, so they are zeroed (untimed, **outside** the timed bracket) before the launch to keep the final NAS verification meaningful.

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record with `time_ms` (GPU event time of the single `gpu_kernel` launch) and `host_time_ms`, then a summary whose `total_time_ms` equals that launch's GPU event time:
```
{"type":"kernel","name":"gpu_kernel","iter":0,"time_ms":...,"host_time_ms":...,"params":{"class":"A","m":28,"block_size":256,"output_sanity":"finite_count_bounds_v1"}}
{"type":"summary","total_time_ms":...}
```

**stderr** — human-readable NAS EP results (Gaussian pair count, Sx/Sy sums, the 10 bin counts) and verification.

## Verification

After the timed launch the host sums the per-block partials into `q[10]`, `Sx`, `Sy` and compares `Sx`/`Sy` against the NAS reference sums for the class (relative error ≤ 1e-8), exactly as in upstream NPB-GPU. If a reference exists for the class and verification fails, the binary exits with a non-zero status.

Before reporting success, every downloaded block's `Sx` and `Sy`, and both
final sums, must be finite. Every histogram bin must be finite, nonnegative and
integral; the sum of a block's bins cannot exceed `NK` times its active thread
count. A partial final block uses its actual active count. All valid counts are
exactly representable in double for the admitted M range. Invalid output or a
failed host output allocation exits nonzero, including for non-reference M.
At most ten invalid-value diagnostics are printed, followed by the total error
count. Host checks and downloads occur after timing; the GPU kernel is unchanged.

`output_sanity=finite_count_bounds_v1` identifies these necessary bounds. They
are **not an independent exact histogram reference**: an incorrect histogram
with legal counts may still pass. The official NAS Sx/Sy criterion is unchanged.
Healthy M values without NAS references still report verification skipped and
are not qualified for performance collection by passing sanity checks.

Regression commands (use a new work directory each time):

```bash
# [TODO for cuda] these tests exercise ../hip/; not yet ported
python3 tests/npb_ep_output_sanity.py --host-only --compiler clang++ --work-dir /tmp/ep-host
python3 tests/npb_ep_output_sanity.py --platform cuda --sweep --work-dir /tmp/ep-device
```

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. The host test exercises the production reduction with fatal ASan/UBSan,
partial blocks, exact limits, invalid counts, nonfinite partials and sum overflow.
The device test corrupts actual output buffers after the timed launch, checks
first/final blocks, bounded diagnostics, both warmer modes, unknown-class skip
behavior and allocation failure. `--sweep` adds all 15 declared M/block-size
combinations with a 60-second per-process limit. CI runs host fixtures and
compiles both production and fault variants; execution requires GPU hardware.

## Platform Notes

- **CUDA**: compiled with `nvcc` from `npb_ep.cu`. Uses double precision (NAS spec); double `atomicAdd` requires sm_60+ (any modern NVIDIA GPU).
