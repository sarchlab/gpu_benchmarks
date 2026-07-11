# torch_compile_bench

Compares PyTorch **eager** vs **torch.compile** inference performance on ResNet-50.

## Setup

```bash
make setup
```

## Run

```bash
make run
```

## Details

- **Model**: ResNet-50 (`torchvision.models.resnet50`, no pretrained weights)
- **Input**: batch of 32 images, 3×224×224 random tensors
- **Mode**: `model.eval()` with `torch.no_grad()`
- **Variants**: eager (default PyTorch) and compiled (`torch.compile`)
- **Timing**: 5 warmup + 5 timed iterations per variant

## Output

CSV to stdout (2 lines):

```
torch_compile_bench,eager,<time_ms>,<images_per_sec>
torch_compile_bench,compiled,<time_ms>,<images_per_sec>
```

Speedup ratio printed to stderr.

## Platform Support

- **CUDA/ROCm**: Full support (bench.py)
