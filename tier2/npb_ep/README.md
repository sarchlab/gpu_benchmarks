# tier2/npb_ep — Embarrassingly Parallel (NAS EP)

Faithful port of the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) **EP** kernel, taken from [NPB-GPU (GMAP)](https://github.com/GMAP/NPB-GPU). The device kernel (`gpu_kernel`) and its random-number generators (`randlc_device`, `vranlc_device`) are the **original upstream source**; only the host driver is restructured for the per-iteration timing harness and JSON output.

EP generates 2^(M+1) pseudorandom numbers, forms Gaussian deviate pairs, and tallies them into 10 concentric square annuli. It is purely compute-bound with negligible memory traffic — the canonical "embarrassingly parallel" workload.

## Algorithm (original NAS EP)

- Generate 2^(M+1) uniform numbers with the NAS 46-bit linear congruential generator `randlc`/`vranlc` (R23/R46 split, `A = 5^13 = 1,220,703,125`, seed `S = 271828183`).
- Each of the 2^(M-16) threads derives its own starting seed via the `randlc` skip-ahead loop, then draws `NK = 2^16` numbers in batches of `RECOMPUTATION = 128`.
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
| `npb_ep.hip` | HIP/CUDA source — original NPB-GPU EP kernel + timing harness |
| `npb_ep.metal` | Metal compute shader — EP kernel |
| `npb_ep_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
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
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Timing model

EP is embarrassingly parallel: **one recorded pass = one `gpu_kernel` launch** (the per-block partial reduction is inside the kernel). Each timed pass emits one `gpu_kernel` record. The RNG seeds are derived from fixed constants inside the kernel, so every launch performs identical work — no input reset is needed. The per-block output accumulators are written via `atomicAdd`, so they are zeroed (untimed, **outside** the timed bracket) before each launch to keep the final NAS verification meaningful.

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per timed pass with `time_ms` (GPU event time of the single `gpu_kernel` launch) and `host_time_ms`, then a summary whose `total_time_ms` is the sum of all passes' GPU event times:
```
{"type":"kernel","name":"gpu_kernel","iter":0,"time_ms":...,"host_time_ms":...,"params":{"class":"A","m":28,"block_size":256}}
...
{"type":"summary","total_time_ms":...}
```

**stderr** — human-readable NAS EP results (Gaussian pair count, Sx/Sy sums, the 10 bin counts) and verification.

## Verification

After the timed loop the host sums the per-block partials into `q[10]`, `Sx`, `Sy` and compares `Sx`/`Sy` against the NAS reference sums for the class (relative error ≤ 1e-8), exactly as in upstream NPB-GPU.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `npb_ep.hip` via `common/hip_cuda_compat.h`. Uses double precision (NAS spec); double `atomicAdd` requires sm_60+ (any modern NVIDIA GPU) or any GCN/CDNA GPU.
- **Metal**: the Metal path (`npb_ep_metal.mm` / `npb_ep.metal`) is a separate implementation and was not updated to the faithful NAS algorithm in this change.
