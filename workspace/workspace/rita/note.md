# Rita - Cycle Notes

## Last completed: Issue #118 - tier5/sdxl_inference
- Created tier5/sdxl_inference/ with bench.py, bench_mps.py, Makefile, README.md
- Committed as 1514fe4 on branch ares/m18-advanced-ml
- SDXL-like UNet config: sample_size=128, block_out_channels=(128,256,512,512), layers_per_block=2, cross_attention_dim=768
- Latent shape: (1, 4, 128, 128), encoder_hidden_states: (1, 77, 768)
- 20 inference steps, DDPMScheduler with 1000 train timesteps
- Metric: steps/sec = 20 / (avg_ms / 1000)
- CSV output: sdxl_inference,20,<avg_ms>,<steps_per_sec>
