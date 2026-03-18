# ResNet-50 Inference Benchmark

PyTorch benchmark: ResNet-50 inference (forward pass only),
batch=64, synthetic [64,3,224,224] input, model.eval(), torch.no_grad(), reports images/sec.

## Requirements

- Python 3.8+
- PyTorch 2.x
- torchvision
- CUDA GPU (NVIDIA) or ROCm GPU (AMD) or Apple Silicon (Metal/MPS)

## Build & Run

### CUDA (NVIDIA)
```
make run PLATFORM=cuda
```

### ROCm (AMD)
```
make run PLATFORM=rocm
```

### Metal (Apple)
```
make run PLATFORM=metal
```

## Output Format

CSV: `benchmark_name,size,time_ms,metric`

Example:
```
resnet50_infer,64,12.345,5184.00
```

## Metric Formula

images/sec = batch_size / (time_ms / 1000)
