# tier2/shoc_devicememory — Device Memory Bandwidth (SHOC)

Benchmarks GPU global memory bandwidth for three access patterns: read-only, write-only, and read-write. Emits per-iteration kernel times; GB/s for each pattern is derived downstream.

Derived from the [SHOC benchmark suite](https://github.com/vetter/shoc) DeviceMemory workload.

## Kernels

| Kernel | Access Pattern | Bandwidth Formula (derived downstream) |
|--------|---------------|-------------------|
| `readOnly_kernel` | Read-only (all threads read; one write/thread) | bytes / time |
| `writeOnly_kernel` | Write-only (strided writes) | bytes / time |
| `readWrite_kernel` | Read + scale + write (in-place) | 2 × bytes / time |

`BENCH_PARAM_transfer_mode` selects which kernels run: `read`, `write`, or
`readwrite` (all three; default).

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_array_size_mb` | 64 | Array size in MB |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_transfer_mode` | readwrite | Access pattern(s): read, write, readwrite |
| `BENCH_PARAM_warmup` | 3 | Warmup launches per kernel (untimed) |
| `BENCH_PARAM_iteration` | 7 | Timed launches per kernel, each recorded |

## Files

| File | Description |
|------|-------------|
| `shoc_devicememory.hip` | HIP source — 3 kernels + self-contained utilities |
| `shoc_devicememory.metal` | Metal compute shaders — 3 kernels |
| `shoc_devicememory_metal.mm` | ObjC++ Metal host code |
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
./shoc_devicememory                        # defaults: 64 MB, readwrite
./shoc_devicememory --array_size_mb 256
./shoc_devicememory --transfer_mode read
```

## Output

**stdout** — JSON lines: for each enabled kernel, one `kernel` record per
timed iteration carrying the GPU event time (`time_ms`) and host wall-clock
time (`host_time_ms`), then a `summary` whose `total_time_ms` is the sum of
all recorded GPU event times:
```
{"type":"kernel","name":"readOnly_kernel","iter":0,"time_ms":2.345600,"host_time_ms":2.400000,"params":{"array_size_mb":64,"block_size":256,"transfer_mode":"readwrite"}}
...
{"type":"kernel","name":"writeOnly_kernel","iter":0,"time_ms":1.987600,"host_time_ms":2.020000,"params":{...}}
...
{"type":"kernel","name":"readWrite_kernel","iter":0,"time_ms":4.123400,"host_time_ms":4.180000,"params":{...}}
...
{"type":"summary","total_time_ms":59.196200}
```

Bandwidth in GB/s is derived downstream from `time_ms` and params.

**stderr** — human-readable per-iteration times:
```
Device: Apple M4 Pro
Array size: 64 MB (16777216 floats)  |  Block size: 256

readOnly_kernel iter 0: gpu = 2.346 ms, host = 2.400 ms
...
Total GPU time: 59.196 ms (7 iterations per kernel)
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `shoc_devicememory.hip`
- **Metal**: compiled with `clang++` from `shoc_devicememory_metal.mm`; the Metal shader is loaded at runtime from the same directory as the binary
