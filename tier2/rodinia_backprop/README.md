# rodinia_backprop — Backpropagation Benchmark

A GPU compute benchmark implementing a two-layer fully-connected neural network
forward and backward pass, based on the Rodinia benchmark suite.

## Algorithm

The benchmark trains a simple two-layer neural network for one iteration:

```
input (INPUT_N) ──[w1, b1]──> hidden (HIDDEN_N) ──[w2, b2]──> output (OUTPUT_N)
```

### Forward Pass
1. **forward_hidden**: `hidden[j] = sigmoid(Σᵢ input[i]·w1[i·HIDDEN_N+j] + b1[j])`
2. **forward_output**: `output[k] = sigmoid(Σⱼ hidden[j]·w2[j·OUTPUT_N+k] + b2[k])`

### Backward Pass
3. **backward_output_delta**: `delta_out[k] = output[k]·(1-output[k])·(target[k]-output[k])`
4. **backward_hidden_delta**: `delta_hid[j] = hidden[j]·(1-hidden[j])·Σₖ w2[j·OUTPUT_N+k]·delta_out[k]`
5. **update_w1**: `w1[i·HIDDEN_N+j] += lr·input[i]·delta_hid[j]`  (2D grid)
6. **update_w2**: `w2[j·OUTPUT_N+k] += lr·hidden[j]·delta_out[k]`

Where `sigmoid(x) = 1 / (1 + e^(-x))` and `lr = 0.1`.

## Default Network Dimensions

| Parameter | Default |
|-----------|---------|
| INPUT_N  (`BENCH_PARAM_input`)  | 65536   |
| HIDDEN_N (`BENCH_PARAM_hidden`) | 1024    |
| OUTPUT_N (`BENCH_PARAM_output`) | 1       |
| `BENCH_PARAM_num_warmup` (warmup epochs, untimed) | 3 |
| `BENCH_PARAM_num_iters` (timed epochs, each recorded) | 7 |

> The w1 weight matrix is `INPUT_N × HIDDEN_N = 65536 × 1024 = 64M floats = 256 MB`.

## Files

| File | Description |
|------|-------------|
| `rodinia_backprop.hip` | HIP/CUDA kernel + host (NVIDIA & AMD) |
| `rodinia_backprop.metal` | Metal compute shaders (Apple) |
| `rodinia_backprop_metal.mm` | Metal Objective-C++ host (Apple) |
| `Makefile` | Build system with auto-detection |
| `README.md` | This file |

## Build

```bash
# Auto-detect platform
make

# Explicit platform
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP
make PLATFORM=metal   # Apple Metal

# Clean
make clean
```

### Requirements

| Platform | Requirement |
|----------|-------------|
| CUDA     | NVIDIA GPU + CUDA Toolkit, `nvcc` |
| ROCm     | AMD GPU + ROCm, `hipcc` (at `/opt/rocm`) |
| Metal    | Apple Silicon or Intel Mac + macOS 10.14+ |

## Run

```bash
./rodinia_backprop                              # defaults
./rodinia_backprop --input 65536 --hidden 1024 # explicit sizes
BENCH_PARAM_num_iters=10 ./rodinia_backprop     # more recorded epochs
```

## Output Format

**stderr** — device info + per-epoch timing details:
```
Device: Apple M2 Pro
Network: 65536 → 1024 → 1  |  Iterations: 3 warmup + 7 timed

epoch 0: gpu = 6.423 ms
...
Total GPU time: 44.961 ms over 7 epochs
```

**stdout** — JSON lines: per timed epoch, one `kernel` record per kernel
(all six kernels), then a `summary` whose `total_time_ms` is the sum of all
recorded GPU event times:
```
{"type":"kernel","name":"forward_hidden","iter":0,"time_ms":5.100000,"host_time_ms":5.160000,"params":{"input":65536,"hidden":1024,"output":1}}
...
{"type":"summary","total_time_ms":44.961000}
```

## GFLOPS Formula

```
FLOPs per epoch ≈ 4 × INPUT_N × HIDDEN_N + 4 × HIDDEN_N × OUTPUT_N
                 (forward + backward, 2 FLOPs per multiply-add)

GFLOPS = FLOPs / (time_s × 10⁹)
```

Derived downstream from the emitted `time_ms` and params; not reported by the binary.

## Notes

- Weights (`w1`, `w2`) initialized uniformly in `[-0.1, 0.1]`
- Biases (`b1`, `b2`) initialized to `0`
- Input initialized as `input[i] = (i % 256) / 256.0`
- Target set to `1.0` for all output units
- Warmup epochs run before the recorded epochs; weight values drift across
  epochs, but every kernel does a fixed amount of work regardless of the
  values, so no state reset is needed between epochs
