# tier2/tango_blackscholes — Black-Scholes European Option Pricing

Benchmarks the Black-Scholes closed-form pricing of European call and put options. Emits one raw kernel time per execution; throughput (M options/sec) is derived downstream.

## Provenance & fidelity

The upstream source and revision for this implementation have not been
established. The previously cited [Tango repository at revision
59db699](https://gitlab.com/Tango-DNNbench/Tango/-/blob/59db69914bf5391b8f39b8b8463482794afe3956/README.md)
describes neural-network inference workloads; it does not establish the origin
of this Black-Scholes implementation. The former claim of a faithful Tango
reproduction was unsupported.

The `tango_blackscholes` benchmark name and `suite: "tango"` manifest value are
retained as existing identifiers, not evidence of upstream provenance. The
complete CPU check below verifies this repository's European call/put formula
with its cumulative-normal approximation. Passing it does not demonstrate
equivalence to a pinned upstream implementation. Source attribution and any
resulting restoration remain open under [R-09 in the readiness
tracker](https://github.com/sarchlab/gpu_benchmarks/issues/71).

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
{"type":"kernel","name":"blackscholes_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.401000,"params":{"size":4194304,"block_size":256,"verification":"full_cpu_v1"}}
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
Black-Scholes verification: PASS (4194304 options, 8388608 prices, 0 mismatching options)
```

## Verification

After timing, the separate scalar CPU reference checks **every measured call
and put price**, recording `verification="full_cpu_v1"`. For each option, both
differences must be at most
`1e-3 * (abs(reference_call) + abs(reference_put)) + 1e-5`; non-finite outputs,
references and tolerances fail. The formula and tolerance are unchanged from
the former 10,000-option prefix check. The diagnostic reports the full option
and price counts, with at most ten mismatch details.

The two reference arrays require `8*N` host bytes (64 MiB at the largest
declared size); their allocation and calculation are outside GPU timing but
inside the process budget. A missing reference allocation reports unavailable
verification and exits nonzero. A mismatch also exits nonzero, so the harness
marks the run as failed while retaining timing records. There is no verification
bypass. This checks the local measured formula; upstream attribution remains
the separate R-09 issue in [the readiness tracker](https://github.com/sarchlab/gpu_benchmarks/issues/71).

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `tango_blackscholes.hip`; one kernel launch, each thread prices one option
