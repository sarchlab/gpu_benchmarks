# SDXL Inference Benchmark

PyTorch benchmark: SDXL-style UNet denoising loop using the diffusers library.
UNet2DConditionModel with SDXL-like config:
sample_size=128, in_channels=4, out_channels=4, layers_per_block=2,
block_out_channels=(128, 256, 512, 512), cross_attention_dim=768.
Runs 20 diffusion denoising steps on a random latent tensor (4x128x128).
Reports steps/sec.

Note: This tests the UNet denoising compute, not the full text-to-image pipeline
(no CLIP text encoder is used).

Model instantiated with random weights — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
- HuggingFace Diffusers
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

CSV: `benchmark_name,steps,time_ms,steps_per_sec`

Example:
```
sdxl_inference,20,1234.567,16.20
```

## Metric Formula

steps/sec = 20 / (avg_time_ms / 1000)
