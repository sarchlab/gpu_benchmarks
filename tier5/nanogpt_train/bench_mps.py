import json
import os
import sys
import time
import torch
import torch.nn as nn
import math

# GPT-2 style model — defined inline (same as bench.py)
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
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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


def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '4'))
    block_size = int(os.environ.get('BENCH_PARAM_block_size', '256'))
    n_layer = int(os.environ.get('BENCH_PARAM_n_layer', '6'))
    n_embd = int(os.environ.get('BENCH_PARAM_n_embd', '384'))
    n_head = int(os.environ.get('BENCH_PARAM_n_head', '6'))
    num_warmup = 0

    vocab_size = 50257

    print(f"NanoGPT train benchmark (MPS) — n_layer={n_layer}, n_embd={n_embd}, "
          f"n_head={n_head}, block_size={block_size}, batch={batch_size}",
          file=sys.stderr)

    model = GPT(vocab_size, n_layer, n_head, n_embd, block_size).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)

    criterion = nn.CrossEntropyLoss()

    def step():
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        return loss

    # Warmup
    for i in range(num_warmup):
        loss = step()
        print(f"  warmup {i+1}/{num_warmup}", file=sys.stderr)
    if num_warmup > 0:
        _ = loss.item()  # sync

    # Timed
    total_time = 0.0
    for i in range(10):
        t0 = time.perf_counter()
        loss = step()
        _ = loss.item()  # force sync
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        print(f"  run {i+1}/10: {elapsed_ms:.3f} ms", file=sys.stderr)

    avg_ms = total_time / 10
    tokens_per_sec = (batch_size * block_size) / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)

    params = {
        "batch_size": batch_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "forward_backward_step",
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
