# rodinia_backprop — Backpropagation Benchmark

Faithful port of the Rodinia Backpropagation benchmark. The two GPU kernels are
the **original** Rodinia 3.1 CUDA source
(`cuda/backprop/backprop_cuda_kernel.cu`); only the host driver is restructured
for single-pass timing.

## Algorithm

The original benchmark trains the input→hidden layer of a 3-layer neural net for
one epoch. On the GPU it runs exactly **two** kernels:

1. **`bpnn_layerforward_CUDA`** — the tiled forward pass. Each block loads a
   `HEIGHT×WIDTH = 16×16` tile of the input→hidden weight matrix and the 16
   corresponding input activations into shared memory, multiplies, and performs
   an **in-block log-step reduction tree** (`for i in 1..log2(HEIGHT)`) to
   produce per-block partial sums.
2. **`bpnn_adjust_weights_cuda`** — the weight/momentum update:
   `w += ETA·δ·y + MOMENTUM·oldw` (with `ETA = MOMENTUM = 0.3`).

The output layer, the partial-sum reduction, and the error/delta computation run
on the **CPU** in the original and are *not* GPU kernels; they are therefore not
timed here, and the `hidden_delta` vector consumed by the adjust kernel is
synthesized rather than back-propagated.

The benchmark executes **one epoch, exactly once** — forward kernel followed by
adjust kernel — and emits one JSON record per kernel. Repetition is the
harness's job (`gpubench run --reps N` re-executes the binary).

## Parameters

| Parameter | Env var | Default | Notes |
|-----------|---------|---------|-------|
| `input` | `BENCH_PARAM_input` (or `--input N`) | 1000000 | input layer size; rounded down to a multiple of 16 |
| `vector_add_warmup` | `BENCH_PARAM_vector_add_warmup` | `true` | cold-cache calibration: boost the clock and evict the LLC before the timed pass |

The hidden layer is **fixed** at 16 units by the `HEIGHT=WIDTH=16` tiling and
`(hid+1)` weight stride — not a knob. Host launch geometry is the original:
`num_blocks = input/16`, `grid(1, num_blocks)`, `threads(16, 16)`.

## Files

| File | Description |
|------|-------------|
| `rodinia_backprop.hip` | HIP/CUDA implementation (faithful kernels + host driver) |
| `Makefile` | Build system with auto-detection |
| `README.md` | This file |

## Build

```bash
make                  # auto-detect platform
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP
make clean
```

## Run

```bash
./rodinia_backprop [--input N]        # N must be a multiple of 16
BENCH_PARAM_input=65536 ./rodinia_backprop
```

## Output

**stdout** — JSON lines: one `kernel` record for each of the two kernels, then a
`summary` whose `total_time_ms` is the sum of the recorded GPU event times:

```
{"type":"kernel","name":"bpnn_layerforward_CUDA","iter":0,"time_ms":...,"host_time_ms":...,"params":{"input":1000000,"hidden":16,"height":16,"width":16}}
{"type":"kernel","name":"bpnn_adjust_weights_cuda","iter":0,"time_ms":...,"host_time_ms":...,"params":{...}}
{"type":"summary","total_time_ms":...}
```

**stderr** — device info and forward/adjust timing.

## Data reset

`bpnn_layerforward_CUDA` overwrites the weight matrix in place, and
`bpnn_adjust_weights_cuda` mutates both the weight and previous-weight (momentum)
buffers. The original re-allocates and re-uploads these buffers on every training
call, so this port re-uploads the weight matrix and momentum buffer (untimed) at
the start of the epoch, and re-uploads the weight matrix again between the two
kernels (the forward kernel clobbers it) — matching the original per-call
semantics. `input` and `hidden_delta` are read-only and uploaded once.

## References

- [Rodinia Benchmark Suite](https://rodinia.cs.virginia.edu/)
- S. Che et al., "Rodinia: A benchmark suite for heterogeneous computing," IISWC 2009
