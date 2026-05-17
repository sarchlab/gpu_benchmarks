import json
import os
import sys
import torch
import torch.nn as nn
import math
import time


# GPT-J style model — defined inline
# Key difference from GPT-2: parallel attention + MLP (not sequential)
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


class GPTJBlock(nn.Module):
    """GPT-J style block: parallel attention + MLP from a single LayerNorm."""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h) + self.mlp(h)
        return x


class GPTJ(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            h=nn.ModuleList([GPTJBlock(n_embd, n_head, block_size)
                             for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(model, prompt, gen_tokens):
    """Autoregressive generation with greedy decoding (argmax)."""
    seq = prompt.clone()
    for _ in range(gen_tokens):
        logits = model(seq)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def benchmark():
    device = torch.device('mps')
    vocab_size = 50257

    # Read parameters from environment variables
    seq_len = int(os.environ.get('BENCH_PARAM_seq_len', '64'))
    n_layer = int(os.environ.get('BENCH_PARAM_n_layer', '12'))
    n_head = int(os.environ.get('BENCH_PARAM_n_head', '16'))
    n_embd = int(os.environ.get('BENCH_PARAM_n_embd', '4096'))
    gen_tokens = int(os.environ.get('BENCH_PARAM_gen_tokens', '64'))
    precision = os.environ.get('BENCH_PARAM_precision', 'fp32')
    num_warmup = 0

    block_size = seq_len + gen_tokens

    print(f"Running gptj_inference (MPS) with seq_len={seq_len}, n_layer={n_layer}, "
          f"n_head={n_head}, n_embd={n_embd}, gen_tokens={gen_tokens}, "
          f"precision={precision}", file=sys.stderr)

    model = GPTJ(vocab_size, n_layer, n_head, n_embd, block_size).to(device)
    model.eval()

    if precision == 'fp16':
        model = model.half()

    prompt = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Warmup
    print("Warmup...", file=sys.stderr)
    for _ in range(num_warmup):
        if precision == 'amp':
            with torch.autocast('mps'):
                out = generate(model, prompt, gen_tokens)
        else:
            out = generate(model, prompt, gen_tokens)
    if num_warmup > 0:
        _ = out[0, -1].item()  # sync

    # Timed: 5 iterations
    print("Benchmarking...", file=sys.stderr)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        if precision == 'amp':
            with torch.autocast('mps'):
                out = generate(model, prompt, gen_tokens)
        else:
            out = generate(model, prompt, gen_tokens)
        _ = out[0, -1].item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    tokens_per_sec = gen_tokens / (avg_ms / 1000)

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Tokens/sec: {tokens_per_sec:.2f}", file=sys.stderr)

    params = {
        "seq_len": seq_len,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "gen_tokens": gen_tokens,
        "precision": precision,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "forward",
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
