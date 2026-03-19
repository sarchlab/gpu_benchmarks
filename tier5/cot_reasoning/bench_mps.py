import torch
import torch.nn as nn
import math
import time
import sys


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


class GPT2Block(nn.Module):
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


class CoTModel(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, hidden_size, max_seq_len):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(
            [GPT2Block(hidden_size, num_heads, max_seq_len) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, num_new_tokens):
        """Autoregressive generation: greedy argmax."""
        seq = input_ids.clone()
        for _ in range(num_new_tokens):
            logits = self.forward(seq)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
        return seq


@torch.no_grad()
def cot_reasoning(model, prompt):
    """Chain-of-Thought reasoning: 3 CoT steps (64 tokens each) + answer (32 tokens)."""
    seq = prompt.clone()
    # Step 1: generate 64 tokens
    seq = model.generate(seq, 64)
    # Step 2: generate 64 tokens
    seq = model.generate(seq, 64)
    # Step 3: generate 64 tokens
    seq = model.generate(seq, 64)
    # Answer: generate 32 tokens
    seq = model.generate(seq, 32)
    return seq


def benchmark():
    device = torch.device('mps')

    # Config
    num_layers = 6
    hidden_size = 384
    num_heads = 6
    vocab_size = 50257
    max_seq_len = 1024
    prompt_len = 32
    total_generated = 3 * 64 + 32  # 224 tokens

    print(f'Chain-of-Thought reasoning benchmark (MPS)', file=sys.stderr)
    print(f'Config: layers={num_layers}, hidden={hidden_size}, heads={num_heads}', file=sys.stderr)
    print(f'Prompt: {prompt_len} tokens, Generated: {total_generated} tokens', file=sys.stderr)

    model = CoTModel(vocab_size, num_layers, num_heads, hidden_size, max_seq_len).to(device)
    model.eval()

    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # Warmup
    print('Warmup...', file=sys.stderr)
    for _ in range(5):
        out = cot_reasoning(model, prompt)
    _ = out[0, -1].item()  # sync

    # Timed
    print('Benchmarking...', file=sys.stderr)
    total_time = 0.0
    for _ in range(10):
        t0 = time.perf_counter()
        out = cot_reasoning(model, prompt)
        _ = out[0, -1].item()  # sync
        t1 = time.perf_counter()
        total_time += (t1 - t0)

    avg_ms = total_time / 10 * 1000
    tokens_per_sec = total_generated / (avg_ms / 1000)
    print(f'Average time: {avg_ms:.3f} ms', file=sys.stderr)
    print(f'Tokens/sec: {tokens_per_sec:.2f}', file=sys.stderr)
    print(f'cot_reasoning,{total_generated},{avg_ms:.3f},{tokens_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
