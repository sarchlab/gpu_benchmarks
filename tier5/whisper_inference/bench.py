import sys
import torch
from transformers import WhisperForConditionalGeneration, WhisperConfig


@torch.no_grad()
def benchmark():
    device = torch.device('cuda')
    config = WhisperConfig(
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024,
        vocab_size=51865,
        max_source_positions=1500,
        max_target_positions=448,
    )
    model = WhisperForConditionalGeneration(config).to(device)
    model.eval()

    batch = 4
    mel_bins = 80
    time_frames = 3000
    num_new_tokens = 64

    # Random mel-spectrogram input
    input_features = torch.randn(batch, mel_bins, time_frames, device=device)
    # Decoder input: start-of-transcript token
    decoder_input_ids = torch.full((batch, 1),
                                   config.decoder_start_token_id or 50258,
                                   dtype=torch.long, device=device)

    print(f"Whisper inference benchmark — encoder_layers={config.encoder_layers}, "
          f"decoder_layers={config.decoder_layers}, d_model={config.d_model}",
          file=sys.stderr)
    print(f"Batch={batch}, mel={mel_bins}x{time_frames}, "
          f"generating {num_new_tokens} tokens", file=sys.stderr)

    def run_generate():
        return model.generate(
            input_features=input_features,
            max_new_tokens=num_new_tokens,
            decoder_input_ids=decoder_input_ids,
        )

    # Warmup
    for i in range(5):
        run_generate()
        print(f"  warmup {i+1}/5", file=sys.stderr)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for i in range(10):
        start.record()
        run_generate()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        total_ms += elapsed
        print(f"  run {i+1}/10: {elapsed:.3f} ms", file=sys.stderr)

    avg_ms = total_ms / 10
    tokens_per_sec = (num_new_tokens * batch) / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)
    print(f'whisper_inference,{batch},{avg_ms:.3f},{tokens_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
