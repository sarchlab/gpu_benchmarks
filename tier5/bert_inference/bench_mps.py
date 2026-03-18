import torch
from transformers import BertModel, BertConfig
import time

def benchmark():
    device = torch.device('mps')
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    model = BertModel(config).to(device)
    model.eval()

    batch = 16
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    _ = out.last_hidden_state.sum().item()  # sync

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            _ = out.last_hidden_state.sum().item()  # force sync
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    ms = sum(times) / len(times)
    sequences_per_sec = batch / (ms / 1000)
    print(f'bert_inference,{batch},{ms:.3f},{sequences_per_sec:.2f}')

if __name__ == '__main__':
    benchmark()
