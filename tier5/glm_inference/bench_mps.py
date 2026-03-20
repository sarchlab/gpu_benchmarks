import json
import os
import sys
import time
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
    device = torch.device('mps')

    # Read parameters from environment variables
    num_new_tokens = int(os.environ.get('BENCH_PARAM_num_new_tokens', '128'))
    prompt_len = int(os.environ.get('BENCH_PARAM_prompt_len', '32'))
    n_layer = int(os.environ.get('BENCH_PARAM_n_layer', '6'))
    n_embd = int(os.environ.get('BENCH_PARAM_n_embd', '512'))
    n_head = int(os.environ.get('BENCH_PARAM_n_head', '8'))

    config = GPT2Config(
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_inner=n_embd * 4,
        vocab_size=32000,
    )
    model = AutoModelForCausalLM.from_config(config).to(device)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"GLM inference benchmark (MPS) — {n_layer} layers, "
          f"hidden={n_embd}, heads={n_head}", file=sys.stderr)
    print(f"Generating {num_new_tokens} tokens from {prompt_len}-token prompt",
          file=sys.stderr)

    # Warmup
    for i in range(5):
        out = generate(model, prompt, num_new_tokens)
        print(f"  warmup {i+1}/5", file=sys.stderr)
    _ = out[0, -1].item()  # sync

    # Timed
    total_time = 0.0
    for i in range(10):
        t0 = time.perf_counter()
        out = generate(model, prompt, num_new_tokens)
        _ = out[0, -1].item()  # force sync
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        print(f"  run {i+1}/10: {elapsed_ms:.3f} ms", file=sys.stderr)

    avg_ms = total_time / 10
    tokens_per_sec = num_new_tokens / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)

    params = {
        "num_new_tokens": num_new_tokens,
        "prompt_len": prompt_len,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
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
