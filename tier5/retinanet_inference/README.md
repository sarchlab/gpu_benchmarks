# RetinaNet Inference Benchmark

PyTorch benchmark: RetinaNet object detection inference (forward pass only) using
RetinaNet-ResNet50-FPN-v2 backbone, synthetic RGB images [B,3,640,640] input,
model.eval(), torch.no_grad(), reports images/sec.

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

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_batch_size` | 4 | Number of images per inference batch |
| `BENCH_PARAM_image_size` | 640 | Spatial resolution of input images |
| `BENCH_PARAM_num_classes` | 91 | Number of object detection classes |
| `BENCH_PARAM_precision` | fp32 | Precision mode (fp32, fp16, amp) |

Example with custom parameters:
```
BENCH_PARAM_batch_size=8 BENCH_PARAM_image_size=800 make run
```

## Output Format

JSON-lines to stdout:

```json
{"type": "kernel", "name": "forward", "time_ms": 45.67, "params": {"batch_size": 4, "image_size": 640, "num_classes": 91, "precision": "fp32"}}
{"type": "summary", "total_time_ms": 45.67, "metrics": [{"name": "images_per_sec", "value": 87.6}]}
```

## Metric Formula

images/sec = batch_size / (time_ms / 1000)
