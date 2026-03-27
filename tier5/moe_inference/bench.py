import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Simple 2-layer FFN expert."""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TopKGating(nn.Module):
    """Linear gate that selects top-k experts per token."""
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        gate_values, indices = torch.topk(probs, self.top_k, dim=-1)
        gate_values = gate_values / gate_values.sum(dim=-1, keepdim=True)
        return gate_values, indices


class MoELayer(nn.Module):
    """Mixture of Experts layer: routes tokens to top-k experts."""
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([Expert(hidden_size) for _ in range(num_experts)])
        self.gating = TopKGating(hidden_size, num_experts, top_k)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        bsz, seq_len, hidden = x.shape
        flat = x.view(-1, hidden)
        gate_values, indices = self.gating(flat)

        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            gate_w = gate_values[:, k].unsqueeze(-1)
            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)
                if mask.any():
                    expert_input = flat[mask]
                    expert_output = self.experts[e_id](expert_input)
                    output[mask] += gate_w[mask] * expert_output

        return output.view(bsz, seq_len, hidden)


class TransformerBlock(nn.Module):
    """Transformer block with MoE feedforward."""
    def __init__(self, hidden_size, num_heads, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.moe = MoELayer(hidden_size, num_experts, top_k)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.moe(h)
        return x


class MoETransformer(nn.Module):
    """MoE Transformer for causal language modeling."""
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 num_experts, top_k, max_seq_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


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
    batch_size = 4

    # Read parameters from environment variables
    num_new_tokens = int(os.environ.get('BENCH_PARAM_num_new_tokens', '64'))
    seq_len = int(os.environ.get('BENCH_PARAM_seq_len', '256'))
    num_experts = int(os.environ.get('BENCH_PARAM_num_experts', '4'))
    top_k = int(os.environ.get('BENCH_PARAM_top_k', '2'))
    num_layers = int(os.environ.get('BENCH_PARAM_num_layers', '4'))
    hidden_size = int(os.environ.get('BENCH_PARAM_hidden_size', '512'))
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    num_heads = max(1, hidden_size // 64)
    max_seq_len = seq_len + num_new_tokens

    model = MoETransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        max_seq_len=max_seq_len,
    ).to(device)
    model.eval()

    prompt = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"MoE inference benchmark — {num_layers} layers, "
          f"hidden={hidden_size}, heads={num_heads}, "
          f"experts={num_experts}, top_k={top_k}", file=sys.stderr)
    print(f"Generating {num_new_tokens} tokens from {seq_len}-token prompt, "
          f"batch={batch_size}", file=sys.stderr)

    # Warmup
    print("Warmup...", file=sys.stderr)
    for i in range(num_warmup):
        generate(model, prompt, num_new_tokens)
        torch.cuda.synchronize()
        print(f"  warmup {i+1}/{num_warmup}", file=sys.stderr)

    # Timed
    print("Benchmarking...", file=sys.stderr)
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

    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec", file=sys.stderr)

    params = {
        "num_new_tokens": num_new_tokens,
        "seq_len": seq_len,
        "num_experts": num_experts,
        "top_k": top_k,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
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
