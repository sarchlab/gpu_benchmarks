import json
import os
import sys
import torch
import time


class DLRM(torch.nn.Module):
    """Deep Learning Recommendation Model (DLRM).

    Architecture: embedding tables (EmbeddingBag) + bottom MLP +
    feature interaction (dot product) + top MLP.
    """

    def __init__(self, num_embeddings, embedding_dim, num_dense_features,
                 num_sparse_features, mlp_bottom_dims, mlp_top_dims):
        super().__init__()

        # Sparse embedding tables (one per sparse feature)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum')
            for _ in range(num_sparse_features)
        ])

        # Bottom MLP: processes dense features
        bottom_layers = []
        in_dim = num_dense_features
        for out_dim in mlp_bottom_dims:
            bottom_layers.append(torch.nn.Linear(in_dim, out_dim))
            bottom_layers.append(torch.nn.ReLU())
            in_dim = out_dim
        # Final bottom layer outputs embedding_dim to match sparse embeddings
        bottom_layers.append(torch.nn.Linear(in_dim, embedding_dim))
        bottom_layers.append(torch.nn.ReLU())
        self.bottom_mlp = torch.nn.Sequential(*bottom_layers)

        # Feature interaction: dot product of all feature vectors
        # Number of interaction features: (num_sparse + 1) choose 2 + embedding_dim
        num_interact_features = num_sparse_features + 1  # sparse + dense
        interact_output_dim = (num_interact_features * (num_interact_features - 1)) // 2 + embedding_dim

        # Top MLP: processes concatenated interaction features
        top_layers = []
        in_dim = interact_output_dim
        for out_dim in mlp_top_dims:
            top_layers.append(torch.nn.Linear(in_dim, out_dim))
            top_layers.append(torch.nn.ReLU())
            in_dim = out_dim
        top_layers.append(torch.nn.Linear(in_dim, 1))
        self.top_mlp = torch.nn.Sequential(*top_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.EmbeddingBag):
                torch.nn.init.xavier_uniform_(m.weight)

    def _feature_interaction(self, dense_out, sparse_outs):
        # Concatenate all feature vectors: [batch, num_features, embedding_dim]
        all_features = torch.cat(
            [dense_out.unsqueeze(1)] + [s.unsqueeze(1) for s in sparse_outs],
            dim=1
        )
        # Dot product interaction: batch matrix multiply
        interaction = torch.bmm(all_features, all_features.transpose(1, 2))
        # Extract upper triangular elements (excluding diagonal)
        batch_size = interaction.size(0)
        triu_indices = torch.triu_indices(
            interaction.size(1), interaction.size(2), offset=1,
            device=interaction.device
        )
        interact_flat = interaction[:, triu_indices[0], triu_indices[1]]
        # Concatenate with dense output
        return torch.cat([dense_out, interact_flat], dim=1)

    def forward(self, dense_input, sparse_offsets, sparse_indices):
        # Bottom MLP on dense features
        dense_out = self.bottom_mlp(dense_input)

        # Embedding lookups for each sparse feature
        sparse_outs = []
        for i, emb in enumerate(self.embeddings):
            sparse_outs.append(emb(sparse_indices[i], sparse_offsets[i]))

        # Feature interaction
        interact_out = self._feature_interaction(dense_out, sparse_outs)

        # Top MLP
        output = self.top_mlp(interact_out)
        return torch.sigmoid(output)


def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '64'))
    num_embeddings = int(os.environ.get('BENCH_PARAM_num_embeddings', '100000'))
    embedding_dim = int(os.environ.get('BENCH_PARAM_embedding_dim', '64'))
    num_dense_features = int(os.environ.get('BENCH_PARAM_num_dense_features', '13'))
    num_sparse_features = int(os.environ.get('BENCH_PARAM_num_sparse_features', '26'))
    precision = os.environ.get('BENCH_PARAM_precision', 'fp32')

    print(f"Running dlrm_inference with batch_size={batch_size}, "
          f"num_embeddings={num_embeddings}, embedding_dim={embedding_dim}, "
          f"num_dense_features={num_dense_features}, "
          f"num_sparse_features={num_sparse_features}, precision={precision}",
          file=sys.stderr)

    # Build model
    mlp_bottom_dims = [256, 128]
    mlp_top_dims = [256, 128]
    model = DLRM(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_dense_features=num_dense_features,
        num_sparse_features=num_sparse_features,
        mlp_bottom_dims=mlp_bottom_dims,
        mlp_top_dims=mlp_top_dims,
    ).to(device)
    model.eval()

    if precision == 'fp16':
        model = model.half()

    # Synthetic data
    dense_input = torch.randn(batch_size, num_dense_features, device=device)
    if precision == 'fp16':
        dense_input = dense_input.half()

    # Sparse inputs: offsets and indices for each embedding table
    lookups_per_sample = 10
    sparse_offsets = []
    sparse_indices = []
    for _ in range(num_sparse_features):
        offsets = torch.arange(
            0, batch_size * lookups_per_sample, lookups_per_sample,
            device=device, dtype=torch.long
        )
        indices = torch.randint(
            0, num_embeddings, (batch_size * lookups_per_sample,),
            device=device, dtype=torch.long
        )
        sparse_offsets.append(offsets)
        sparse_indices.append(indices)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            if precision == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(dense_input, sparse_offsets, sparse_indices)
            else:
                _ = model(dense_input, sparse_offsets, sparse_indices)
    torch.cuda.synchronize()

    num_iters = 5

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(num_iters):
            if precision == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(dense_input, sparse_offsets, sparse_indices)
            else:
                _ = model(dense_input, sparse_offsets, sparse_indices)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / num_iters
    samples_per_sec = batch_size / (avg_ms / 1000)

    params = {
        "batch_size": batch_size,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "num_dense_features": num_dense_features,
        "num_sparse_features": num_sparse_features,
        "precision": precision,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "forward",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "samples_per_sec", "value": samples_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
