import sys
import time
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
    device = torch.device('mps')

    # SDXL-like UNet config for benchmarking (no model download)
    unet = UNet2DConditionModel(
        sample_size=128,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,
    ).to(device)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    batch = 1
    num_steps = 20

    # Random latent and text encoder hidden states
    latents = torch.randn(batch, 4, 128, 128, device=device)
    encoder_hidden_states = torch.randn(batch, 77, 768, device=device)

    print(f"SDXL UNet denoising benchmark — "
          f"block_out_channels=(128,256,512,512), layers_per_block=2",
          file=sys.stderr)
    print(f"Batch={batch}, latent=4x128x128, {num_steps} diffusion steps",
          file=sys.stderr)

    # Warmup
    for i in range(5):
        noisy = torch.randn_like(latents)
        result = denoise_loop(unet, scheduler, noisy, encoder_hidden_states,
                              num_steps)
        print(f"  warmup {i+1}/5", file=sys.stderr)
    _ = result[0, 0, 0, 0].item()  # sync

    # Timed
    total_time = 0.0
    for i in range(10):
        noisy = torch.randn_like(latents)
        t0 = time.perf_counter()
        result = denoise_loop(unet, scheduler, noisy, encoder_hidden_states,
                              num_steps)
        _ = result[0, 0, 0, 0].item()  # force sync
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        print(f"  run {i+1}/10: {elapsed_ms:.3f} ms", file=sys.stderr)

    avg_ms = total_time / 10
    steps_per_sec = num_steps / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {steps_per_sec:.2f} steps/sec",
          file=sys.stderr)
    print(f'sdxl_inference,{num_steps},{avg_ms:.3f},{steps_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
