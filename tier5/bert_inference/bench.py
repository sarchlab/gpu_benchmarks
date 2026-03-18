import torch
from transformers import BertModel, BertConfig
import time

def benchmark():
    device = torch.device('cuda')
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
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 10
    sequences_per_sec = batch / (ms / 1000)
    print(f'bert_inference,{batch},{ms:.3f},{sequences_per_sec:.2f}')

if __name__ == '__main__':
    benchmark()
