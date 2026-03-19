import sys
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig


def benchmark():
    device = torch.device('cuda')
    config = GPTNeoXConfig(
        num_hidden_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        rotary_pct=0.25,
        vocab_size=50257,
        max_position_embeddings=2048,
    )
    model = GPTNeoXForCausalLM(config).to(device)
    model.train()

    batch_size = 4
    seq_len = 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len),
                              device=device)
    labels = input_ids.clone()

    print(f"GPT-NeoX training benchmark — {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, heads={config.num_attention_heads}",
          file=sys.stderr)
    print(f"Batch={batch_size}, seq_len={seq_len}, "
          f"forward+backward+optimizer step", file=sys.stderr)

    # Warmup
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  warmup {i+1}/5", file=sys.stderr)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for i in range(10):
        start.record()
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        total_ms += elapsed
        print(f"  run {i+1}/10: {elapsed:.3f} ms", file=sys.stderr)

    avg_ms = total_ms / 10
    tokens_per_sec = (batch_size * seq_len) / (avg_ms / 1000)
    print(f"  avg: {avg_ms:.3f} ms, {tokens_per_sec:.2f} tokens/sec",
          file=sys.stderr)
    print(f'gptneox_train,{batch_size},{avg_ms:.3f},{tokens_per_sec:.2f}')


if __name__ == '__main__':
    benchmark()
