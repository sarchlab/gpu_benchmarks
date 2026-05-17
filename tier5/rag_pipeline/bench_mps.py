import json
import os
import sys
import time
import torch
import torch.nn as nn
import math


class Encoder(nn.Module):
    """BERT-like encoder: embedding + positional + transformer encoder layers + mean pooling."""

    def __init__(self, vocab_size=50257, hidden=256, heads=4, num_layers=4, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.transformer(x)
        # Mean pooling over sequence dimension
        return x.mean(dim=1)  # (B, hidden)


class Retriever(nn.Module):
    """Retrieves top-k documents by cosine similarity."""

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, query_emb, doc_emb_matrix):
        # query_emb: (B, hidden), doc_emb_matrix: (num_docs, hidden)
        query_norm = nn.functional.normalize(query_emb, dim=-1)
        doc_norm = nn.functional.normalize(doc_emb_matrix, dim=-1)
        similarity = torch.matmul(query_norm, doc_norm.t())  # (B, num_docs)
        topk_indices = similarity.topk(self.top_k, dim=-1).indices  # (B, top_k)
        # Gather top-k document embeddings
        topk_embs = doc_emb_matrix[topk_indices]  # (B, top_k, hidden)
        return topk_indices, topk_embs


class Generator(nn.Module):
    """GPT-2-like generator with cross-attention to retrieved context."""

    def __init__(self, vocab_size=50257, hidden=256, heads=4, num_layers=4, max_len=512):
        super().__init__()
        self.hidden = hidden
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, start_token_ids, context, generate_len=64):
        B = start_token_ids.shape[0]
        generated = start_token_ids  # (B, 1)
        for _ in range(generate_len):
            S = generated.shape[1]
            positions = torch.arange(S, device=generated.device).unsqueeze(0).expand(B, S)
            x = self.token_emb(generated) + self.pos_emb(positions)
            # Causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=generated.device)
            x = self.transformer(x, context, tgt_mask=causal_mask)
            logits = self.head(x[:, -1:, :])  # (B, 1, vocab)
            next_token = logits.argmax(dim=-1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated  # (B, 1 + generate_len)


class RAGPipeline(nn.Module):
    """Combines Encoder + Retriever + Generator."""

    def __init__(self, vocab_size=50257, hidden=256, heads=4, num_layers=4, top_k=5):
        super().__init__()
        self.encoder = Encoder(vocab_size=vocab_size, hidden=hidden, heads=heads, num_layers=num_layers)
        self.retriever = Retriever(top_k=top_k)
        self.generator = Generator(vocab_size=vocab_size, hidden=hidden, heads=heads, num_layers=num_layers)

    def forward(self, query_ids, doc_emb_matrix, generate_len=64):
        # Encode queries
        query_emb = self.encoder(query_ids)  # (B, hidden)
        # Retrieve top-k docs
        _, topk_embs = self.retriever(query_emb, doc_emb_matrix)  # (B, top_k, hidden)
        # Generate tokens using retrieved context as cross-attention memory
        start_tokens = torch.zeros(query_ids.shape[0], 1, dtype=torch.long, device=query_ids.device)
        output = self.generator(start_tokens, topk_embs, generate_len=generate_len)
        return output


@torch.no_grad()
def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    num_docs = int(os.environ.get('BENCH_PARAM_num_docs', '1000'))
    batch = int(os.environ.get('BENCH_PARAM_batch', '4'))
    top_k = int(os.environ.get('BENCH_PARAM_top_k', '5'))
    generate_len = int(os.environ.get('BENCH_PARAM_generate_len', '64'))
    num_layers = int(os.environ.get('BENCH_PARAM_num_layers', '4'))
    hidden = int(os.environ.get('BENCH_PARAM_hidden', '256'))
    num_warmup = 0

    # Fixed config
    vocab_size = 50257
    heads = 4
    doc_len = 128
    query_len = 32

    print(f"RAG Pipeline Benchmark (MPS)", file=sys.stderr)
    print(f"  encoder/generator: layers={num_layers}, hidden={hidden}, heads={heads}", file=sys.stderr)
    print(f"  num_docs={num_docs}, doc_len={doc_len}, top_k={top_k}", file=sys.stderr)
    print(f"  batch={batch}, query_len={query_len}, generate_len={generate_len}", file=sys.stderr)

    # Build model
    model = RAGPipeline(
        vocab_size=vocab_size, hidden=hidden, heads=heads,
        num_layers=num_layers, top_k=top_k
    ).to(device)
    model.eval()

    # Precompute document embeddings
    print("Precomputing document embeddings...", file=sys.stderr)
    doc_embs_list = []
    doc_batch_size = 50
    for i in range(0, num_docs, doc_batch_size):
        end = min(i + doc_batch_size, num_docs)
        doc_ids = torch.randint(0, vocab_size, (end - i, doc_len), device=device)
        emb = model.encoder(doc_ids)
        doc_embs_list.append(emb)
    doc_emb_matrix = torch.cat(doc_embs_list, dim=0)  # (num_docs, hidden)
    _ = doc_emb_matrix.sum().item()  # sync
    print(f"Document embeddings shape: {doc_emb_matrix.shape}", file=sys.stderr)

    # Prepare query input
    query_ids = torch.randint(0, vocab_size, (batch, query_len), device=device)

    # Warmup
    print("Warming up...", file=sys.stderr)
    for _ in range(num_warmup):
        out = model(query_ids, doc_emb_matrix, generate_len=generate_len)
    if num_warmup > 0:
        _ = out.sum().item()  # sync

    # Timed iterations
    print("Benchmarking...", file=sys.stderr)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = model(query_ids, doc_emb_matrix, generate_len=generate_len)
        _ = out.sum().item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    queries_per_sec = batch / (avg_ms / 1000)

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Queries/sec: {queries_per_sec:.2f}", file=sys.stderr)

    params = {
        "num_docs": num_docs,
        "batch": batch,
        "top_k": top_k,
        "generate_len": generate_len,
        "num_layers": num_layers,
        "hidden": hidden,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "rag_pipeline",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "queries_per_sec", "value": queries_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
