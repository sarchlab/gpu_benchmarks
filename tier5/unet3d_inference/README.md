# 3D U-Net Inference Benchmark

PyTorch benchmark: 3D U-Net inference (forward pass only) for medical image segmentation.
Encoder-decoder architecture with skip connections, 3D convolutions, batch normalization, and ReLU.
Synthetic 3D volumes (e.g., [B, 1, 128, 128, 128]), Xavier-initialized weights, model.eval(), torch.no_grad().
Reports volumes/sec throughput.

## Requirements

- Python 3.8+
- PyTorch 2.x
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

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_batch_size` | 2 | Number of 3D volumes per batch |
| `BENCH_PARAM_volume_depth` | 128 | Depth of input volume |
| `BENCH_PARAM_volume_height` | 128 | Height of input volume |
| `BENCH_PARAM_volume_width` | 128 | Width of input volume |
| `BENCH_PARAM_in_channels` | 1 | Number of input channels |
| `BENCH_PARAM_num_classes` | 3 | Number of segmentation classes |
| `BENCH_PARAM_precision` | fp32 | Precision mode (fp32, fp16, amp) |

## Output Format

JSON-lines to stdout:

```json
{"type": "kernel", "name": "forward", "time_ms": 123.45, "params": {...}}
{"type": "summary", "total_time_ms": 123.45, "metrics": [{"name": "volumes_per_sec", "value": 16.2}]}
```

## Metric Formula

volumes/sec = batch_size / (time_ms / 1000)
