import sys
import torch
from transformers import AutoModelForCausalLM, GPT2Config


@torch.no_grad()
def generate(model, prompt, num_new_tokens):
    """Autoregressive generation: generate num_new_tokens starting from prompt."""
    seq = prompt.clone()
    for _ in range(num_new_tokens):
        logits = model(seq).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def benchmark():
    device = torch.device('cuda')
    # GLM-style config using GPT2Config as base (GLM not always available
    # as a standalone config in all transformers versions)
    config = GPT2Config(
        n_layer=6,
        n_embd=512,
        n_head=8,
        n_inner=2048,
        vocab_size=32000,
    )
    model = AutoModelForCausalLM.from_config(config).to(device)
    model.eval()

    prompt_len = 32
    num_new_tokens = 128
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"GLM inference benchmark — {config.n_layer} layers, "
          f"hidden={config.n_embd}, heads={config.n_head}", file=sys.stderr)
    print(f"Generating {num_new_tokens} tokens from {prompt_len}-token prompt",
          file=sys.stderr)

    # Warmup
    for i in range(5):
        generate(model, prompt, num_new_tokens)
        print(f"  warmup {i+1}/5", file=sys.stderr)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for i in range(10):
        start.record()
        generate(model, prompt, num_new_tokens)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        total_ms += elapsed
        print(f"  run {i+1}/10: {elapsed:.3f} ms", file=sys.stderr)

    avg_ms = total_ms / 10
    tokens_per_sec = num_new_tokens / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)
    print(f'glm_inference,{num_new_tokens},{avg_ms:.3f},{tokens_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
