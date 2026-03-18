import torch
import torch.nn as nn
import math
import time

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
    vocab_size = 50257
    n_layer, n_head, n_embd, block_size = 6, 6, 384, 256
    batch = 4

    model = GPT(vocab_size, n_layer, n_head, n_embd, block_size).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, vocab_size, (batch, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch, block_size), device=device)

    criterion = nn.CrossEntropyLoss()

    def step():
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        return loss

    # Warmup
    for _ in range(5):
        loss = step()
    _ = loss.item()  # sync

    # Timed
    total_time = 0.0
    t_start = time.perf_counter()
    for _ in range(10):
        step()
    loss = step()
    _ = loss.item()  # sync (after all steps done, sync once)
    # Redo properly with per-step sync
    total_time = 0.0
    for _ in range(10):
        t0 = time.perf_counter()
        loss = step()
        _ = loss.item()
        t1 = time.perf_counter()
        total_time += (t1 - t0)

    ms_per_step = total_time / 10 * 1000
    tokens_per_sec = batch * block_size * 10 / total_time
    print(f'nanogpt_train,{batch * block_size},{ms_per_step:.3f},{tokens_per_sec:.2f}')

if __name__ == '__main__':
    benchmark()
