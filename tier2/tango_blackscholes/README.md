# tier2/tango_blackscholes — Black-Scholes European Option Pricing

Benchmarks the Black-Scholes closed-form pricing of European call and put options. Emits one raw kernel time per execution; throughput (M options/sec) is derived downstream.

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

## Measurement protocol

The pricing kernel is launched and timed **exactly once** per execution — there
is no in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup`
is `true` (default), the vector-add warmer runs immediately before the timed
launch: ~100 ms of sustained load ramps the GPU clock while its final pass
evicts the last-level cache, so the kernel is measured cold-cache at a boosted
clock. Repetition is the harness's job: `gpubench run --reps N` re-executes the
binary in fresh processes.

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_size` | 4194304 | Number of options to price |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Files

| File | Description |
|------|-------------|
| `tango_blackscholes.hip` | HIP source — Black-Scholes kernel (one thread per option) |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./tango_blackscholes                              # defaults: N=4194304
./tango_blackscholes --size 1048576               # fewer options
BENCH_PARAM_size=1048576 ./tango_blackscholes     # env vars override argv
```

## Output

**stdout** — JSON lines: one kernel record with the GPU event time (`time_ms`)
and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms`
equals that record's GPU event time:
```
{"type":"kernel","name":"blackscholes_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.401000,"params":{"size":4194304,"block_size":256}}
{"type":"summary","total_time_ms":2.345600}
```

Derived metrics such as M options/sec are computed downstream from `time_ms`
and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Options: 4194304

blackscholes_kernel: gpu = 2.346 ms, host = 2.401 ms
Total GPU time: 2.346 ms
PASS
```

## Verification

After the timed run (measurements already emitted), GPU results are compared
against a CPU reference for the first 10,000 options — both call and put
prices, with relative tolerance. On FAIL the process exits non-zero, so the
harness marks the run as failed without losing the measurements. There is no
verification on/off parameter.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `tango_blackscholes.hip`; one kernel launch, each thread prices one option
