# tier2/tango_blackscholes — Black-Scholes European Option Pricing

Benchmarks the Black-Scholes closed-form pricing of European call and put options. Emits raw per-iteration kernel times; throughput (M options/sec) is derived downstream.

Derived from the [Tango benchmark suite](https://github.com/AkshayUsha/Tango) Black-Scholes workload.

## Algorithm

- **N** options (default: N=4,194,304 = 4M)
- Each thread prices one option, computing both call and put prices
- Uses the Black-Scholes formula:
  - **d1** = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
  - **d2** = d1 − σ·√T
  - **Call** = S·N(d1) − K·e^(−rT)·N(d2)
  - **Put** = K·e^(−rT)·(1−N(d2)) − S·(1−N(d1))
- Cumulative normal distribution N(x) approximated via Abramowitz & Stegun polynomial (26.2.17)
- Input parameters randomized: S∈[5,200], K∈[1,300], T∈[0.25,10], σ∈[0.1,1.0], r=0.02
- M options/sec = N / time\_sec / 10⁶ (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `tango_blackscholes.hip` | HIP source — Black-Scholes kernel (one thread per option) |
| `tango_blackscholes.metal` | Metal compute shader — Black-Scholes kernel |
| `tango_blackscholes_metal.mm` | ObjC++ Metal host code |
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
./tango_blackscholes                              # defaults: N=4194304
./tango_blackscholes --size 1048576               # fewer options
BENCH_PARAM_size=1048576 ./tango_blackscholes     # env vars override argv
```

Environment variables: `BENCH_PARAM_size`, `BENCH_PARAM_block_size`,
`BENCH_PARAM_precision`, plus `BENCH_PARAM_num_warmup` (warmup launches,
default 3) and `BENCH_PARAM_num_iters` (timed launches, each recorded,
default 7).

## Output

**stdout** — JSON lines, one kernel record per timed iteration with the GPU
event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"blackscholes_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.401000,"params":{"size":4194304,"block_size":256,"precision":"float"}}
...
{"type":"summary","total_time_ms":16.419200}
```

Derived metrics such as M options/sec are computed downstream from `time_ms`
and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Options: 4194304  |  Iterations: 3 warmup + 7 timed

blackscholes_kernel iter 0: gpu = 2.346 ms, host = 2.401 ms
...
Total GPU time: 16.419 ms over 7 iterations
PASS
```

## Verification

- Compares GPU results against CPU reference for the first 10,000 options
- Checks both call and put prices with relative tolerance

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `tango_blackscholes.hip`; one kernel launch, each thread prices one option
- **Metal**: compiled with `clang++` from `tango_blackscholes_metal.mm`; the Metal shader (`tango_blackscholes.metal`) is loaded and compiled at runtime from the same directory as the binary
