import json
import os
import sys
import torch
import torch.nn as nn
import math
import time


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
def cot_reasoning(model, prompt, cot_step_tokens, answer_tokens):
    """Chain-of-Thought reasoning: 3 CoT steps + answer generation."""
    seq = prompt.clone()
    # Step 1
    seq = model.generate(seq, cot_step_tokens)
    # Step 2
    seq = model.generate(seq, cot_step_tokens)
    # Step 3
    seq = model.generate(seq, cot_step_tokens)
    # Answer
    seq = model.generate(seq, answer_tokens)
    return seq


def benchmark():
    device = torch.device('mps')
    vocab_size = 50257

    # Read parameters from environment variables
    total_generated = int(os.environ.get('BENCH_PARAM_total_generated', '224'))
    prompt_len = int(os.environ.get('BENCH_PARAM_prompt_len', '32'))
    num_layers = int(os.environ.get('BENCH_PARAM_num_layers', '6'))
    hidden_size = int(os.environ.get('BENCH_PARAM_hidden_size', '384'))
    num_heads = int(os.environ.get('BENCH_PARAM_num_heads', '6'))

    # Split total_generated into 3 CoT steps + answer
    cot_step_tokens = total_generated // 4
    answer_tokens = total_generated - 3 * cot_step_tokens
    max_seq_len = prompt_len + total_generated

    print(f"Chain-of-Thought reasoning benchmark (MPS)", file=sys.stderr)
    print(f"Config: layers={num_layers}, hidden={hidden_size}, heads={num_heads}", file=sys.stderr)
    print(f"Prompt: {prompt_len} tokens, Generated: {total_generated} tokens "
          f"(3x{cot_step_tokens} + {answer_tokens})", file=sys.stderr)

    model = CoTModel(vocab_size, num_layers, num_heads, hidden_size, max_seq_len).to(device)
    model.eval()

    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # Warmup
    print("Warmup...", file=sys.stderr)
    for _ in range(5):
        out = cot_reasoning(model, prompt, cot_step_tokens, answer_tokens)
    _ = out[0, -1].item()  # sync

    # Timed
    print("Benchmarking...", file=sys.stderr)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = cot_reasoning(model, prompt, cot_step_tokens, answer_tokens)
        _ = out[0, -1].item()  # sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    tokens_per_sec = total_generated / (avg_ms / 1000)

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Tokens/sec: {tokens_per_sec:.2f}", file=sys.stderr)

    params = {
        "total_generated": total_generated,
        "prompt_len": prompt_len,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "reason",
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
