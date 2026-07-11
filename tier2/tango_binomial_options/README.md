# tier2/tango_binomial_options — Binomial Option Pricing (CRR Model)

Benchmarks the Cox-Ross-Rubinstein (CRR) binomial tree model for **American put option** pricing with early exercise. Emits one raw kernel time per execution; throughput (options/sec) is derived downstream.

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

## Measurement protocol

The binomial kernel is launched and timed **exactly once** per execution —
there is no in-program warmup or iteration loop. When
`BENCH_PARAM_vector_add_warmup` is `true` (default), the vector-add warmer runs
immediately before the timed launch: ~100 ms of sustained load ramps the GPU
clock while its final pass evicts the last-level cache, so the kernel is
measured cold-cache at a boosted clock. Repetition is the harness's job:
`gpubench run --reps N` re-executes the binary in fresh processes.

## Files

| File | Description |
|------|-------------|
| `tango_binomial_options.hip` | HIP source — binomial tree kernel (one block per option, shared memory) |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./tango_binomial_options                                  # defaults: 512 options, 1024 steps
./tango_binomial_options --options 256 --steps 512        # fewer options and steps
```

### Environment variables (override argv)

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_options` | 512 | Number of options |
| `BENCH_PARAM_steps` | 1024 | Binomial tree steps per option (clamped to 4096) |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` equals that record's GPU event time:
```
{"type":"kernel","name":"binomial_kernel","iter":0,"time_ms":45.6789,"host_time_ms":45.9012,"params":{"options":512,"steps":1024}}
{"type":"summary","total_time_ms":45.678900}
```

Derived metrics (options/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Options: 512  |  Steps: 1024

binomial_kernel: gpu = 45.679 ms, host = 45.901 ms
Total GPU time: 45.679 ms
PASS
```

## Verification

- After the timed run (measurements already emitted), GPU results are compared against a CPU reference for the first 10 options
- CPU reference uses identical backward induction algorithm
- Tolerance: relative error with absolute floor
- On FAIL the process exits non-zero, so the harness marks the run as failed without losing the measurements; there is no verification on/off parameter

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `tango_binomial_options.hip`; one block per option, shared memory for tree levels, `__syncthreads()` between levels
