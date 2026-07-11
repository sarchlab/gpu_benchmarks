# tier3/cudnn_conv — Forward 2D Convolution via cuDNN / MIOpen

A multi-platform benchmark for forward 2D convolution, wrapping vendor-optimised
deep learning libraries:

| Platform | Library                              |
|----------|--------------------------------------|
| NVIDIA   | **cuDNN** (`cudnnConvolutionForward`) |
| AMD      | **MIOpen** (`miopenConvolutionForward`) |

## Convolution Parameters

| Parameter         | Value     |
|-------------------|-----------|
| Batch size (N)    | 32        |
| Input channels (C)| 3         |
| Input size (H×W)  | 224 × 224 |
| Output channels (K)| 64       |
| Kernel size (R×S) | 7 × 7     |
| Stride            | 2         |
| Padding           | 3         |
| Output size       | 112 × 112 |

This configuration matches the first convolution layer of ResNet-50 / VGG-style
networks operating on ImageNet-sized (224×224) RGB images.

## Algorithm

Computes forward convolution in **NCHW** layout:

```
output[n][k][oh][ow] = Σ_{c,r,s} input[n][c][oh*stride-pad+r][ow*stride-pad+s] × filter[k][c][r][s]
```

### cuDNN / MIOpen

- Descriptors are created for input tensor, output tensor, filter, and convolution
- Algorithm auto-tuning selects the fastest available kernel
- A workspace buffer is allocated for algorithms that require temporary memory

## Files

| File                     | Description                                           |
|--------------------------|-------------------------------------------------------|
| `cudnn_conv.hip`         | CUDA/ROCm source (`#ifdef USE_CUDNN` / `USE_MIOPEN`) |
| `Makefile`               | Auto-detecting build system                           |
| `README.md`              | This file                                             |

## Build

Auto-detect (recommended):

```bash
make
```

Explicit platform:

```bash
make PLATFORM=cuda    # NVIDIA CUDA + cuDNN
make PLATFORM=rocm    # AMD ROCm/HIP + MIOpen
```

Clean:

```bash
make clean
```

## Run

```bash
./cudnn_conv
```

## Output Format

The benchmark emits two kinds of output:

**CSV (stdout)** — machine-parseable:

```
cudnn_conv,32x3x224x224,<time_ms>,<GFLOPS>
```

Example:

```
cudnn_conv,32x3x224x224,5.678,1330.42
```

**Human-readable (stderr):**

```
cuDNN algo selected: 1 (of 8 candidates)
Workspace: 1048576 bytes
Verification (1×4×4×4 patch): PASS
cudnn_conv  N=32 C=3 H=224 W=224 K=64 R=7 S=7 stride=2 pad=3  time_ms=5.678  GFLOPS=1330.42
```

## Performance Metric

```
GFLOPS = 2 × N × K × H_out × W_out × C × R × S / time_s / 1e9
```

The factor of 2 accounts for one multiply and one add per element of the
convolution accumulation.

## Verification

After the timed run, a small patch of the GPU output (batch=0, first 4 output
channels, 4×4 spatial region) is compared against a CPU reference convolution.
The relative error threshold is 1e-3.

The benchmark exits with status 0 on pass and 1 on failure.

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the average

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).
