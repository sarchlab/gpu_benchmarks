# tier2/npb_ep — Embarrassingly Parallel (NAS EP)

Benchmarks the generation of **N pairs of Gaussian random deviates** using the Box-Muller transform with a per-thread linear congruential RNG. Counts results into 10 annular bins. Purely compute-bound workload with minimal memory traffic.

Derived from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) EP kernel.

## Algorithm

- **N** pairs of Gaussian random deviates (default: N=16,777,216 = 2²⁴)
- Per-thread linear congruential RNG (a = 5¹³ = 1,220,703,125, mod 2³²)
- Box-Muller transform: uniform (u₁, u₂) → Gaussian (x₁, x₂)
  - r = √(−2 ln u₁), θ = 2π u₂
  - x₁ = r cos θ, x₂ = r sin θ
- Bin index = floor(√(x₁² + x₂²)), capped at 9 (10 bins total)
- Atomic increment of bin counters

## Files

| File | Description |
|------|-------------|
| `npb_ep.hip` | HIP source — EP kernel with LCG + Box-Muller + binning |
| `npb_ep.metal` | Metal compute shader — EP kernel |
| `npb_ep_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./npb_ep                              # defaults: N=16777216
./npb_ep --size 1048576               # smaller workload
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 16777216 | Number of Gaussian pairs |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_seed` | 271828183 | RNG seed parameter (recorded in params) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup launches (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed launches, each recorded |

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per timed iteration with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"ep_kernel","iter":0,"time_ms":8.2345,"host_time_ms":8.3011,"params":{"size":16777216,...}}
...
{"type":"summary","total_time_ms":57.6415}
```

Derived metrics (Gpairs/sec) are computed downstream from `time_ms` and the params. Bin counters are reset (untimed) before every launch so each iteration performs identical work.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
EP pairs: 16777216 (2^24)  |  Iterations: 3 warmup + 7 timed

ep_kernel iter 0: gpu = 8.234 ms, host = 8.301 ms
...
Total GPU time: 57.642 ms over 7 iterations

Bin counts (GPU vs CPU, N=1048576):
  Bin 0: GPU=697254  CPU=697254
  Bin 1: GPU=315462  CPU=315462
  ...
PASS
```

## Verification

- Runs CPU reference EP with same RNG and compares bin counts exactly
- For large N, verification uses a capped subset (N ≤ 1,048,576) for tractable CPU time

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `npb_ep.hip`; one kernel launch, atomicAdd for bin counting
- **Metal**: compiled with `clang++` from `npb_ep_metal.mm`; the Metal shader (`npb_ep.metal`) is loaded and compiled at runtime from the same directory as the binary
