# tier2/shoc_devicememory — Device Memory Bandwidth (SHOC)

Benchmarks GPU global memory bandwidth for three access patterns: read-only, write-only, and read-write. Emits one raw per-kernel GPU time per execution; GB/s for each pattern is derived downstream.

Derived from the [SHOC benchmark suite](https://github.com/vetter/shoc) DeviceMemory workload.

## Kernels

| Kernel | Access Pattern | Bandwidth Formula (derived downstream) |
|--------|---------------|-------------------|
| `readOnly_kernel` | Read-only (all threads read; one write/thread) | bytes / time |
| `writeOnly_kernel` | Write-only (strided writes) | bytes / time |
| `readWrite_kernel` | Read + scale + write (in-place) | 2 × bytes / time |

`BENCH_PARAM_transfer_mode` selects which kernels run: `read`, `write`, or
`readwrite` (all three; default).

## Measurement protocol

Each enabled kernel is launched and timed **exactly once** per execution —
there is no in-program warmup or iteration loop. When
`BENCH_PARAM_vector_add_warmup` is `true` (default), the vector-add warmer runs
immediately before each timed launch: ~100 ms of sustained load ramps the GPU
clock while its final pass evicts the last-level cache, so every kernel is
measured cold-cache at a boosted clock. Repetition is the harness's job:
`gpubench run --reps N` re-executes the binary in fresh processes.

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_array_size_mb` | 64 | Array size in MB |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_transfer_mode` | readwrite | Access pattern(s): read, write, readwrite |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before each timed launch |

## Files

| File | Description |
|------|-------------|
| `shoc_devicememory.hip` | HIP source — 3 kernels + self-contained utilities |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./shoc_devicememory                        # defaults: 64 MB, readwrite
./shoc_devicememory --array_size_mb 256
./shoc_devicememory --transfer_mode read
```

## Output

**stdout** — JSON lines: one `kernel` record per enabled kernel carrying the
GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
`summary` whose `total_time_ms` is the sum of the recorded GPU event times
(with `transfer_mode=readwrite`, all three kernels):
```
{"type":"kernel","name":"readOnly_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.400000,"params":{"array_size_mb":64,"block_size":256,"transfer_mode":"readwrite"}}
{"type":"kernel","name":"writeOnly_kernel","iter":0,"time_ms":1.987600,"host_time_ms":2.020000,"params":{...}}
{"type":"kernel","name":"readWrite_kernel","iter":0,"time_ms":4.123400,"host_time_ms":4.180000,"params":{...}}
{"type":"summary","total_time_ms":8.456600}
```

Bandwidth in GB/s is derived downstream from `time_ms` and params.

**stderr** — human-readable per-kernel times:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Array size: 64 MB (16777216 floats)  |  Block size: 256

readOnly_kernel: gpu = 2.346 ms, host = 2.400 ms
writeOnly_kernel: gpu = 1.988 ms, host = 2.020 ms
readWrite_kernel: gpu = 4.123 ms, host = 4.180 ms
Total GPU time: 8.457 ms
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `shoc_devicememory.hip`
