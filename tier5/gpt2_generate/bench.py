import json
import os
import sys
import torch
import torch.nn as nn
import math


# GPT-2 style model — defined inline
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            h=nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(model, prompt, num_new_tokens):
    """Autoregressive generation: generate num_new_tokens starting from prompt."""
    seq = prompt.clone()
    for _ in range(num_new_tokens):
        logits = model(seq)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def benchmark():
    device = torch.device('cuda')
    vocab_size = 50257

    # Read parameters from environment variables
    num_new_tokens = int(os.environ.get('BENCH_PARAM_num_new_tokens', '128'))
    prompt_len = int(os.environ.get('BENCH_PARAM_prompt_len', '32'))
    n_layer = int(os.environ.get('BENCH_PARAM_n_layer', '6'))
    n_embd = int(os.environ.get('BENCH_PARAM_n_embd', '384'))
    n_head = int(os.environ.get('BENCH_PARAM_n_head', '6'))

    block_size = prompt_len + num_new_tokens

    print(f"Running gpt2_generate with num_new_tokens={num_new_tokens}, "
          f"prompt_len={prompt_len}, n_layer={n_layer}, n_embd={n_embd}, "
          f"n_head={n_head}", file=sys.stderr)

    model = GPT(vocab_size, n_layer, n_head, n_embd, block_size).to(device)
    model.eval()

    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # Warmup
    print("Warmup...", file=sys.stderr)
    for _ in range(5):
        generate(model, prompt, num_new_tokens)
    torch.cuda.synchronize()

    # Timed
    print("Benchmarking...", file=sys.stderr)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(5):
        start.record()
        generate(model, prompt, num_new_tokens)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    avg_ms = total_ms / 5
    tokens_per_sec = num_new_tokens / (avg_ms / 1000)

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Tokens/sec: {tokens_per_sec:.2f}", file=sys.stderr)

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
