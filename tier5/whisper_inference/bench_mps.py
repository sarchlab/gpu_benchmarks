import json
import os
import sys
import time
import torch
from transformers import WhisperForConditionalGeneration, WhisperConfig


@torch.no_grad()
def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    num_new_tokens = int(os.environ.get('BENCH_PARAM_num_new_tokens', '64'))
    batch = int(os.environ.get('BENCH_PARAM_batch', '4'))
    time_frames = int(os.environ.get('BENCH_PARAM_time_frames', '3000'))
    encoder_layers = int(os.environ.get('BENCH_PARAM_encoder_layers', '4'))
    d_model = int(os.environ.get('BENCH_PARAM_d_model', '256'))
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    config = WhisperConfig(
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=encoder_layers,
        encoder_attention_heads=max(1, d_model // 64),
        decoder_attention_heads=max(1, d_model // 64),
        encoder_ffn_dim=d_model * 4,
        decoder_ffn_dim=d_model * 4,
        vocab_size=51865,
        max_source_positions=1500,
        max_target_positions=448,
    )
    model = WhisperForConditionalGeneration(config).to(device)
    model.eval()

    mel_bins = 80

    # Random mel-spectrogram input
    input_features = torch.randn(batch, mel_bins, time_frames, device=device)
    # Decoder input: start-of-transcript token
    decoder_input_ids = torch.full((batch, 1),
                                   config.decoder_start_token_id or 50258,
                                   dtype=torch.long, device=device)

    print(f"Whisper inference benchmark (MPS) — encoder_layers={encoder_layers}, "
          f"d_model={d_model}, batch={batch}, time_frames={time_frames}",
          file=sys.stderr)
    print(f"Generating {num_new_tokens} tokens", file=sys.stderr)

    def run_generate():
        return model.generate(
            input_features=input_features,
            max_new_tokens=num_new_tokens,
            decoder_input_ids=decoder_input_ids,
        )

    # Warmup
    for i in range(num_warmup):
        out = run_generate()
        print(f"  warmup {i+1}/{num_warmup}", file=sys.stderr)
    if num_warmup > 0:
        _ = out[0, -1].item()  # sync

    # Timed
    total_time = 0.0
    for i in range(10):
        t0 = time.perf_counter()
        out = run_generate()
        _ = out[0, -1].item()  # force sync
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        print(f"  run {i+1}/10: {elapsed_ms:.3f} ms", file=sys.stderr)

    avg_ms = total_time / 10
    tokens_per_sec = (num_new_tokens * batch) / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)

    params = {
        "num_new_tokens": num_new_tokens,
        "batch": batch,
        "time_frames": time_frames,
        "encoder_layers": encoder_layers,
        "d_model": d_model,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "generate",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "tokens_per_sec", "value": tokens_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
