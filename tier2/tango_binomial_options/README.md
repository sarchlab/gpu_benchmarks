# tier2/tango_binomial_options — Binomial Option Pricing (CRR Model)

Benchmarks the Cox-Ross-Rubinstein (CRR) binomial tree model for **American put option** pricing with early exercise. Reports per-iteration GPU times; throughput (options/sec) is derived downstream.

Derived from the [Tango benchmark suite](https://github.com/AkshayUsha/Tango) binomial options workload.

## Algorithm

- **N** options (default: N=512), each with **S** steps (default: S=1024)
- Each threadblock/threadgroup processes one option:
  1. **Terminal payoffs**: compute put payoff max(K − S·u^j·d^(S−j), 0) at each leaf node
  2. **Backward induction**: walk backwards through S tree levels, at each node computing max(discounted continuation value, early exercise value)
- Uses shared/threadgroup memory for the option value array at each level
- CRR parameters: u = e^(σ√Δt), d = 1/u, p = (e^(rΔt) − d) / (u − d)
- Input parameters randomized: S∈[5,200], K∈[1,300], T∈[0.25,10], σ∈[0.1,1.0], r=0.02
- Metric: options/sec = N / time\_sec (derived downstream from the recorded times)

## Files

| File | Description |
|------|-------------|
| `tango_binomial_options.hip` | HIP source — binomial tree kernel (one block per option, shared memory) |
| `tango_binomial_options.metal` | Metal compute shader — binomial tree kernel (threadgroup memory) |
| `tango_binomial_options_metal.mm` | ObjC++ Metal host code |
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
./tango_binomial_options                                  # defaults: 512 options, 1024 steps
./tango_binomial_options --options 256 --steps 512        # fewer options and steps
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_options` | 512 | Number of options |
| `BENCH_PARAM_steps` | 1024 | Binomial tree steps per option |
| `BENCH_PARAM_precision` | float | Precision label (float, double, half; recorded in params) |
| `BENCH_PARAM_warmup` | 3 | Warmup launches (untimed) |
| `BENCH_PARAM_iteration` | 7 | Timed launches, each recorded |

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per timed iteration with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"binomial_kernel","iter":0,"time_ms":45.6789,"host_time_ms":45.9012,"params":{"options":512,"steps":1024,"precision":"float"}}
...
{"type":"summary","total_time_ms":319.7523}
```

Derived metrics (options/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Options: 512  |  Steps: 1024  |  Iterations: 3 warmup + 7 timed

binomial_kernel iter 0: gpu = 45.679 ms, host = 45.901 ms
...
Total GPU time: 319.752 ms over 7 iterations
PASS
```

## Verification

- Compares GPU results against CPU reference for the first 10 options
- CPU reference uses identical backward induction algorithm
- Tolerance: relative error with absolute floor

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `tango_binomial_options.hip`; one block per option, shared memory for tree levels, `__syncthreads()` between levels
- **Metal**: compiled with `clang++` from `tango_binomial_options_metal.mm`; the Metal shader (`tango_binomial_options.metal`) is loaded and compiled at runtime from the same directory as the binary; uses threadgroup memory and `threadgroup_barrier`
