# common_cuda — shared headers for native CUDA benchmarks

Native CUDA counterparts of the headers in `common/`, for benchmarks under
`tier*/<bench>/cuda/`. They follow the same measurement contract and JSON output
protocol as `common/`, but call the CUDA runtime API directly, with no
compatibility layer.

| File | Counterpart | Contents |
|------|-------------|----------|
| `bench_timing.h` | `common/bench_timing.h` | event/chrono timing brackets, `bench_time_op`, JSON emitters |
| `bench_warmup.h` | `common/bench_warmup.h` | cold-cache clock warmer (768 MB scratch, ~100 ms sustained load) |

Keep both trees in sync when changing the measurement methodology.
