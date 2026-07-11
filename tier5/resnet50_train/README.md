# ResNet-50 Training Benchmark

PyTorch benchmark: ResNet-50 training step (forward + backward + optimizer step),
batch=32, synthetic [32,3,224,224] input, SGD optimizer, reports images/sec.

## Requirements

- Python 3.8+
- PyTorch 2.x
- torchvision
- CUDA GPU (NVIDIA) or ROCm GPU (AMD)

## Build & Run

### CUDA (NVIDIA)
```
make run PLATFORM=cuda
```

### ROCm (AMD)
```
make run PLATFORM=rocm
```

## Output Format

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
resnet50_train,32,45.123,710.23
```

## Metric Formula

images/sec = batch_size / (time_ms / 1000)
