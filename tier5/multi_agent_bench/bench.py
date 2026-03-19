import torch
import torch.nn as nn
import math
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


class AgentModel(nn.Module):
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
def multi_agent_pipeline(agent_a, agent_b, agent_c, prompt):
    """Multi-agent pipeline: 3 agents each generate 64 tokens."""
    # Agent A: starts with prompt (32 tokens), generates 64 tokens
    output_a = agent_a.generate(prompt, 64)

    # Agent B: takes last 64 tokens of output_a as input, generates 64 tokens
    input_b = output_a[:, -64:]
    output_b = agent_b.generate(input_b, 64)

    # Agent C: takes concat of output_a[-32:] + output_b[-32:] (64 tokens), generates 64 tokens
    input_c = torch.cat([output_a[:, -32:], output_b[:, -32:]], dim=1)
    output_c = agent_c.generate(input_c, 64)

    return output_c


def benchmark():
    device = torch.device('cuda')

    # Config
    num_layers = 4
    hidden_size = 256
    num_heads = 4
    vocab_size = 50257
    max_seq_len = 512
    prompt_len = 32
    total_generated = 3 * 64  # 192 tokens

    print(f'Multi-agent LLM benchmark', file=sys.stderr)
    print(f'Config: layers={num_layers}, hidden={hidden_size}, heads={num_heads}', file=sys.stderr)
    print(f'Agents: 3, Tokens per agent: 64, Total: {total_generated}', file=sys.stderr)

    agent_a = AgentModel(vocab_size, num_layers, num_heads, hidden_size, max_seq_len).to(device)
    agent_b = AgentModel(vocab_size, num_layers, num_heads, hidden_size, max_seq_len).to(device)
    agent_c = AgentModel(vocab_size, num_layers, num_heads, hidden_size, max_seq_len).to(device)
    agent_a.eval()
    agent_b.eval()
    agent_c.eval()

    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # Warmup
    print('Warmup...', file=sys.stderr)
    for _ in range(5):
        multi_agent_pipeline(agent_a, agent_b, agent_c, prompt)
    torch.cuda.synchronize()

    # Timed
    print('Benchmarking...', file=sys.stderr)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(10):
        start.record()
        multi_agent_pipeline(agent_a, agent_b, agent_c, prompt)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    avg_ms = total_ms / 10
    tokens_per_sec = total_generated / (avg_ms / 1000)
    print(f'Average time: {avg_ms:.3f} ms', file=sys.stderr)
    print(f'Tokens/sec: {tokens_per_sec:.2f}', file=sys.stderr)
    print(f'multi_agent_bench,3x64,{avg_ms:.3f},{tokens_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
