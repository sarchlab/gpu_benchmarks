# tier2/tango_binomial_options — Binomial Option Pricing (CRR Model)

Benchmarks the Cox-Ross-Rubinstein (CRR) binomial tree model for **American put option** pricing with early exercise. Emits one raw kernel time per execution; throughput (options/sec) is derived downstream.

## Provenance & fidelity

Derived from the [Tango benchmark suite](https://gitlab.com/Tango-DNNbench/Tango) — the binomial options workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Prices American puts with a per-node `powf` vs the upstream European FMA induction — a transcendental-heavy, different workload class. (A shared-memory race in the backward induction has been fixed.)

Read this as "a binomial-options benchmark in the spirit of Tango," not Tango's binomial options.

## Algorithm

- **N** options (default: N=512), each with a fixed **S=1024** steps
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
| `cpu_reference.h` | Complete CPU price reference with cached payoffs and bounded parallelism |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./tango_binomial_options                                  # defaults: 512 options
./tango_binomial_options --options 256                    # fewer options
```

The binomial-tree resolution is fixed at the canonical **1024 steps** and is not
a parameter: it enters the work as steps², so sweeping it would only multiply
runtime (into timeout territory at large `options`) without producing a distinct
workload. It is still emitted in each record as `"steps":1024`.

### Environment variables (override argv)

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_options` | 512 | Number of options |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record with `time_ms` (GPU event time) and `host_time_ms` (host wall-clock), then a summary whose `total_time_ms` equals that record's GPU event time:
```
{"type":"kernel","name":"binomial_kernel","iter":0,"time_ms":45.6789,"host_time_ms":45.9012,"params":{"options":512,"steps":1024,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":45.678900}
```

Derived metrics (options/sec) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Options: 512  |  Steps: 1024

binomial_kernel: gpu = 45.679 ms, host = 45.901 ms
Total GPU time: 45.679 ms
Binomial verification: PASS (512 prices, 0 mismatches)
```

## Verification

After timing and output emission, every measured option price is downloaded and
checked against the existing CPU American-put recurrence. Both values must be
finite and their absolute difference must be at most
`1e-2 * abs(CPU price) + 1e-4`. Each kernel record identifies this policy as
`verification="full_cpu_v1"`. Diagnostics report the complete price/mismatch
counts and print details for at most ten mismatches. A mismatch or unavailable
full CPU reference causes a nonzero process exit; previously emitted timing
records remain available to the harness. There is no verification-off mode.

The reference caches the identical payoff expression for every exponent in
`-1024..1024`, preserving the original continuation arithmetic. An explicit
NaN-aware maximum permits vectorization without fast-math. At most 16 CPU
workers evaluate independent options, with a sequential fallback if a thread
cannot be created. Reference storage is one float per option (4 MiB at the
largest declared size), plus approximately 12 KiB of scratch per active worker.
All reference work occurs after the timed GPU launch. The kernel, input sequence,
33 declared sizes and 60-second collection budget are unchanged.

The process budget includes CPU verification. On the qualified H100 host using
16 CPU workers, the largest declared size completes in about 26 seconds end to
end. Fewer available CPU workers or host contention can exceed that budget;
supported host/size profiles remain part of R-17/R-22 in the readiness tracker.

The host regression compares against the original per-node-power recurrence,
including early exercise, extreme generated inputs, complete parallel results
and NaN/infinity selection. The measured-result regression corrupts actual GPU
prices before their production download, including outputs beyond the former
ten-price prefix and the largest-size tail. See [test instructions](../../tests/README.md).
This repair verifies the current local formula; it does not settle upstream
attribution under [R-09](https://github.com/sarchlab/gpu_benchmarks/issues/71),
general input/capacity admission or AMD hardware qualification.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `tango_binomial_options.hip`; one block per option, shared memory for tree levels, `__syncthreads()` between levels
