import json
import os
import sys
import torch
from transformers import PhiForCausalLM, PhiConfig


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

    # Read parameters from environment variables
    num_new_tokens = int(os.environ.get('BENCH_PARAM_num_new_tokens', '128'))
    prompt_len = int(os.environ.get('BENCH_PARAM_prompt_len', '32'))
    num_hidden_layers = int(os.environ.get('BENCH_PARAM_num_hidden_layers', '6'))
    hidden_size = int(os.environ.get('BENCH_PARAM_hidden_size', '512'))
    num_attention_heads = int(os.environ.get('BENCH_PARAM_num_attention_heads', '8'))
    num_warmup = 0

    config = PhiConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        vocab_size=51200,
    )
    model = PhiForCausalLM(config).to(device)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"Phi inference benchmark — {num_hidden_layers} layers, "
          f"hidden={hidden_size}, heads={num_attention_heads}",
          file=sys.stderr)
    print(f"Generating {num_new_tokens} tokens from {prompt_len}-token prompt",
          file=sys.stderr)

    # Warmup
    for i in range(num_warmup):
        generate(model, prompt, num_new_tokens)
        print(f"  warmup {i+1}/{num_warmup}", file=sys.stderr)
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

    params = {
        "num_new_tokens": num_new_tokens,
        "prompt_len": prompt_len,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
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
