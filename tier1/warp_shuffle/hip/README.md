# warp_shuffle — dependent native-wave shuffle chains

Each thread starts with its global integer thread ID and executes
`ops_per_thread` dependent shuffles. Every thread stores its final value in a
separate output element. One selected variant runs once per process, producing
one GPU event time and host launch/synchronization time. Repetitions are fresh
processes (`gpubench run --reps N`).

| Variant | Intrinsic | Source lane on each iteration |
|---------|-----------|-------------------------------|
| `shfl_xor` | `__shfl_xor(val, 1, warpSize)` | Lane XOR 1 |
| `shfl` | `__shfl(val, (threadIdx.x ^ 1) % warpSize, warpSize)` | Same pair exchange, using the indexed intrinsic |
| `shfl_up` | `__shfl_up(val, 1, warpSize)` | Previous lane; lane 0 retains its value |
| `shfl_down` | `__shfl_down(val, 1, warpSize)` | Next lane; the last lane retains its value |

XOR and indexed cases have the same data dependency but exercise different
intrinsics. Up/down values propagate toward a boundary and eventually become
uniform within a wave. A boundary lane returning its own value still invokes
the intrinsic; a counted shuffle does not imply a distinct useful exchange.
The generated instruction sequence and active-lane counts require qualification
for each compiler/target. A source dependency alone is not proof that every
instruction survives optimization.

## Build and run

```sh
cd tier1/warp_shuffle/hip
make PLATFORM=cuda       # nvcc -x cu; default architecture: native
# make PLATFORM=rocm     # hipcc; native runtime wave width
./warp_shuffle          # 4096 operations, one block, 256 threads, shfl_xor
./warp_shuffle 8192
BENCH_PARAM_shuffle_variant=all ./warp_shuffle
```

The Makefile can auto-detect ROCm from `/opt/rocm`, otherwise CUDA. Both builds
use shared headers from `../../../common_hip`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `BENCH_PARAM_ops_per_thread` (or argv[1]) | 4096 | Positive number of intrinsic invocations per thread |
| `BENCH_PARAM_num_blocks` | 1 | Grid blocks |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block; complete native waves only |
| `BENCH_PARAM_shuffle_variant` | `shfl_xor` | One table variant, or manual `all` |
| `BENCH_PARAM_vector_add_warmup` | `true` | Optional clock/cache warmer before each timed variant |

The Cartesian manifest includes only the four individual variants: 29 operation
counts × four block sizes × four variants = 464 configurations. `all` remains
available directly in the binary for a four-variant run; it is excluded from the
sweep because those launches would duplicate the individual cases. The default
variant is now `shfl_xor`, matching the manifest. Measurement controls are fixed
per collection session, not extra Cartesian axes.

The warmer aims to raise clocks and evict the last-level cache; its achieved
state needs target verification. Setting it to `false` runs each variant as-is.
There are no untimed shuffle launches. The per-thread final stores are part of
the timed kernel; allocation and test verification are outside that interval.

## Native width and supported geometry

Kernels use the device `warpSize` builtin, and output records contain the selected
device's `warp_size` property. Code does not assume a width based on vendor:
NVIDIA uses 32 lanes; AMD's native runtime width depends on architecture
(for example, gfx9 uses 64 and gfx10+ uses 32). This follows the
[HIP porting guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html).

The binary admits native width 32 or 64, positive dimensions within device limits,
and a block size divisible by that width. This ensures all source lanes are
active, including the full-mask CUDA compatibility intrinsics. It rejects
incomplete waves, invalid variants, and dimensions that cannot represent the
per-thread allocation or signed integer IDs before allocating or timing work.
It does not round dimensions or silently select another variant.

Do not force an alternate wave width unsupported by the HIP runtime. In
particular, AMD documents that RDNA `-mwavefrontsize64` is unsupported by HIP;
see [HIP language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html).
A CUDA run does not qualify AMD wave32 or wave64 execution. Those native targets
must separately pass output and instruction checks under R-19 before admission
to a reference campaign.

## Output and interpretation

A default run emits two JSON lines (times below are illustrative):

```json
{"type":"kernel","name":"shfl_xor_kernel","iter":0,"time_ms":0.02,"host_time_ms":0.04,"params":{"ops_per_thread":4096,"num_blocks":1,"threads_per_block":256,"shuffle_variant":"shfl_xor","warp_size":32}}
{"type":"summary","total_time_ms":0.02}
```

Each manual `all` record retains `"shuffle_variant":"all"`; its kernel name
identifies the intrinsic. The summary sums the selected launches' GPU times.
For each individual record, the downstream logical rate is
`num_blocks * threads_per_block * ops_per_thread / (time_ms * 1e-3)`.
Multiplying by four bytes describes logical register-value movement, not memory
bandwidth or a measured physical link rate.

The dependent chain is a latency-sensitive workload. Its event interval also
includes launch/event overhead, loop instructions, and one final store per
thread. Multiple waves may hide dependency latency. Neither event time divided
by operation count nor the aggregate logical rate isolates an instruction's
latency or establishes peak shuffle throughput. The default one-block rate
cannot establish a device peak by multiplication with SM/CU count. See the
[measurement interpretation guide](../../../docs/measurement_interpretation.md).

## Functional qualification

```sh
make PLATFORM=cuda test  # or PLATFORM=rocm; requires a compatible GPU
# From the repository root, using the production parser/acceptance gate:
GPUBENCH_WARP_SHUFFLE_BINARY="$PWD/tier1/warp_shuffle/hip/warp_shuffle" \
  go test ./cmd/gpubench -run '^TestWarpShuffleOutput$' -count=1
# Add GPUBENCH_GPU_SWEEP=1 for every declared operation count.
```

The output test invokes the production kernels and checks every thread against
closed-form lane origins, with allocation guards. Odd/even and minimum/maximum
counts cover pair swapping, up/down boundaries and complete native waves across
multiple blocks. CI compiles this GPU test; it cannot run it without a GPU.
The Go check covers the four sweep variants plus the manual `all` compatibility
path, every declared block size, and records the native width. Both are
functional checks with no performance claim. Inspect target instructions and
active-lane counts separately; correct final values alone cannot prove how much
work the compiler emitted.
