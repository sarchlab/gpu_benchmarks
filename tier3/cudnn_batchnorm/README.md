# tier3/cudnn_batchnorm — Batch Normalization via cuDNN / MIOpen / MPS

A multi-platform benchmark for batch normalization (forward + backward),
wrapping vendor-optimised DNN libraries:

| Platform | Library                            | Operations        |
|----------|------------------------------------|--------------------|
| NVIDIA   | **cuDNN** (`cudnnBatchNormalization*`)   | Forward + Backward |
| AMD      | **MIOpen** (`miopenBatchNormalization*`) | Forward + Backward |
| Apple    | **Metal Performance Shaders** (`MPSCNNBatchNormalization`) | Forward only |

## Algorithm

Computes batch normalization (spatial mode):

**Forward Training:**
```
y = scale * (x - mean) / sqrt(var + epsilon) + bias
```

**Backward:**
```
grad_input  = f(grad_output, x, scale, saved_mean, saved_inv_var)
grad_scale  = sum(grad_output * (x - mean) / sqrt(var + eps))
grad_bias   = sum(grad_output)
```

### Parameters

- Input:  **N=64, C=256, H=W=56** (NCHW layout, FP32)
- Scale/Bias: per-channel (C=256 elements each)
- Running mean/variance: per-channel, exponential average factor = 0.1
- Epsilon: 1e-5
- Mode: `CUDNN_BATCHNORM_SPATIAL` / `miopenBNSpatial`

### Metal Note

The Apple Metal version uses `MPSCNNBatchNormalization` for **forward-only**
inference. The backward pass (`MPSCNNBatchNormalizationGradient`) requires a
more complex training graph setup via `MPSNNGraph` and is not included.
The bandwidth metric for Metal reflects forward-only data movement.

## Files

| File                           | Description                                      |
|--------------------------------|--------------------------------------------------|
| `cudnn_batchnorm.hip`          | CUDA/ROCm source (`#ifdef USE_CUDNN` / `#elif defined(USE_MIOPEN)`) |
| `cudnn_batchnorm_metal.mm`     | Objective-C++ Metal host using `MPSCNNBatchNormalization` |
| `Makefile`                     | Auto-detecting build system                      |
| `README.md`                    | This file                                        |

## Build

Auto-detect (recommended):

```bash
make
```

Explicit platform:

```bash
make PLATFORM=cuda    # NVIDIA CUDA + cuDNN
make PLATFORM=rocm    # AMD ROCm/HIP + MIOpen
make PLATFORM=metal   # Apple Metal + MPS
```

Clean:

```bash
make clean
```

## Run

```bash
./cudnn_batchnorm
```

## Output Format

The benchmark emits two kinds of output:

**CSV (stdout)** — machine-parseable:

```
cudnn_batchnorm,64x256x56x56,<time_ms>,<GBs>
```

Example:

```
cudnn_batchnorm,64x256x56x56,1.234,500.00
```

**Human-readable (stderr):**

```
cudnn_batchnorm  N=64 C=256 H=56 W=56  time_ms=1.234  GB/s=500.00
```

## Performance Metric

```
Effective bandwidth (GB/s) = total_bytes / time_s / 1e9
```

For forward + backward (CUDA/ROCm):
```
total_bytes ≈ 3 × N × C × H × W × sizeof(float) × 2
```

This accounts for:
- **Forward:** read input (N×C×H×W), write output (N×C×H×W)
- **Backward:** read input (N×C×H×W), read grad_output (N×C×H×W), write grad_input (N×C×H×W)
- Plus per-channel scale/bias/mean/var (negligible relative to spatial data)

For forward-only (Metal):
```
total_bytes ≈ 2 × N × C × H × W × sizeof(float)
```

## Verification

After the timed run, a sanity check verifies that:
1. Output values are not NaN or Inf
2. Output is not all zeros

The benchmark exits with status 0 on pass and 1 on failure.

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the average

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).
On Metal, timing uses `mach_absolute_time()` around `waitUntilCompleted`.
