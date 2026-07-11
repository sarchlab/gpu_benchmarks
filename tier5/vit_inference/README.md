# ViT-B/16 Inference Benchmark

PyTorch benchmark: Vision Transformer (ViT-B/16) inference (forward pass),
batch=32, synthetic [32,3,224,224] input, model.eval(), torch.no_grad(), reports images/sec.

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
vit_inference,32,18.234,1755.00
```

## Metric Formula

images/sec = batch_size / (time_ms / 1000)
