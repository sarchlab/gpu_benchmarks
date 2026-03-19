# Stable Diffusion Benchmark

PyTorch benchmark: Stable Diffusion UNet denoising loop using the diffusers library.
UNet2DConditionModel with small random-weight config:
sample_size=64, in_channels=4, out_channels=4, layers_per_block=1,
block_out_channels=(64, 128, 256), cross_attention_dim=256.
Runs 20 diffusion denoising steps on a random latent tensor (4x64x64).
Reports steps/sec.

Note: This tests the UNet denoising compute, not the full text-to-image pipeline
(no CLIP text encoder is used).

Model instantiated with random weights — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
- HuggingFace Diffusers
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

CSV: `benchmark_name,steps,time_ms,steps_per_sec`

Example:
```
stable_diffusion_bench,20,456.789,43.78
```

## Metric Formula

steps/sec = 20 / (avg_time_ms / 1000)
