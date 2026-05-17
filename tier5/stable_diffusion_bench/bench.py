import json
import os
import sys
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler


@torch.no_grad()
def denoise_loop(unet, scheduler, latents, encoder_hidden_states, num_steps):
    """Run the denoising loop for num_steps."""
    scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    num_steps = int(os.environ.get('BENCH_PARAM_num_steps', '20'))
    batch = int(os.environ.get('BENCH_PARAM_batch', '1'))
    sample_size = int(os.environ.get('BENCH_PARAM_sample_size', '64'))
    layers_per_block = int(os.environ.get('BENCH_PARAM_layers_per_block', '1'))
    cross_attention_dim = int(os.environ.get('BENCH_PARAM_cross_attention_dim', '256'))
    num_warmup = 0

    # Small UNet config for benchmarking (no model download)
    unet = UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=4,
        out_channels=4,
        layers_per_block=layers_per_block,
        block_out_channels=(64, 128, 256),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=cross_attention_dim,
    ).to(device)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Random latent and text encoder hidden states
    latents = torch.randn(batch, 4, sample_size, sample_size, device=device)
    encoder_hidden_states = torch.randn(batch, 77, cross_attention_dim, device=device)

    print(f"Stable Diffusion UNet denoising benchmark — "
          f"block_out_channels=(64,128,256), layers_per_block={layers_per_block}",
          file=sys.stderr)
    print(f"Batch={batch}, latent=4x{sample_size}x{sample_size}, {num_steps} diffusion steps, "
          f"cross_attention_dim={cross_attention_dim}",
          file=sys.stderr)

    # Warmup
    for i in range(num_warmup):
        noisy = torch.randn_like(latents)
        denoise_loop(unet, scheduler, noisy, encoder_hidden_states, num_steps)
        print(f"  warmup {i+1}/{num_warmup}", file=sys.stderr)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for i in range(10):
        noisy = torch.randn_like(latents)
        start.record()
        denoise_loop(unet, scheduler, noisy, encoder_hidden_states, num_steps)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        total_ms += elapsed
        print(f"  run {i+1}/10: {elapsed:.3f} ms", file=sys.stderr)

    avg_ms = total_ms / 10
    steps_per_sec = num_steps / (avg_ms / 1000)
    images_per_sec = batch / (avg_ms / 1000)

    print(f"  avg: {avg_ms:.3f} ms, {steps_per_sec:.2f} steps/sec, "
          f"{images_per_sec:.2f} images/sec", file=sys.stderr)

    params = {
        "num_steps": num_steps,
        "batch": batch,
        "sample_size": sample_size,
        "layers_per_block": layers_per_block,
        "cross_attention_dim": cross_attention_dim,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "stable_diffusion_denoise",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "steps_per_sec", "value": steps_per_sec},
            {"name": "images_per_sec", "value": images_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
