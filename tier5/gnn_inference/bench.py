import json
import os
import sys
import torch
import time


class GATLayer(torch.nn.Module):
    """Single Graph Attention layer (multi-head) implemented in pure PyTorch."""

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = torch.nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = torch.nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]  (src, dst)
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Linear projection: [num_nodes, num_heads, head_dim]
        h = self.W(x).view(num_nodes, self.num_heads, self.head_dim)

        # Attention scores per edge
        # a_src contribution for source nodes, a_dst for destination nodes
        score_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # [num_nodes, num_heads]
        score_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [num_nodes, num_heads]

        # Edge attention: e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        edge_attn = self.leaky_relu(
            score_src[src] + score_dst[dst]
        )  # [num_edges, num_heads]

        # Softmax per destination node using scatter
        # Subtract max for numerical stability
        attn_max = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attn_max.scatter_reduce_(
            0,
            dst.unsqueeze(1).expand(-1, self.num_heads),
            edge_attn,
            reduce='amax',
            include_self=True,
        )
        edge_attn = edge_attn - attn_max[dst]
        edge_attn = edge_attn.exp()

        # Sum of exp for normalization
        attn_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attn_sum.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, self.num_heads),
            edge_attn,
        )
        edge_attn = edge_attn / (attn_sum[dst] + 1e-8)

        # Aggregate: weighted sum of source features per destination
        # [num_edges, num_heads, head_dim]
        msg = h[src] * edge_attn.unsqueeze(-1)

        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim),
            msg,
        )

        # Reshape: [num_nodes, out_dim]
        return out.view(num_nodes, -1)


class RGAT(torch.nn.Module):
    """R-GAT proxy model: multi-layer Graph Attention Network."""

    def __init__(self, in_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # First layer
        self.layers.append(GATLayer(in_dim, hidden_dim, num_heads))
        self.norms.append(torch.nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, num_heads))
            self.norms.append(torch.nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = torch.nn.functional.elu(x)
        return x


def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    num_nodes = int(os.environ.get('BENCH_PARAM_num_nodes', '10000'))
    num_edges = int(os.environ.get('BENCH_PARAM_num_edges', '50000'))
    hidden_dim = int(os.environ.get('BENCH_PARAM_hidden_dim', '128'))
    num_heads = int(os.environ.get('BENCH_PARAM_num_heads', '4'))
    num_layers = int(os.environ.get('BENCH_PARAM_num_layers', '3'))
    precision = os.environ.get('BENCH_PARAM_precision', 'fp32')

    print(f"Running gnn_inference with num_nodes={num_nodes}, num_edges={num_edges}, "
          f"hidden_dim={hidden_dim}, num_heads={num_heads}, num_layers={num_layers}, "
          f"precision={precision}", file=sys.stderr)

    # Build model
    in_dim = hidden_dim
    model = RGAT(in_dim, hidden_dim, num_heads, num_layers).to(device)
    model.eval()

    if precision == 'fp16':
        model = model.half()

    # Synthetic graph data
    x = torch.randn(num_nodes, in_dim, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    if precision == 'fp16':
        x = x.half()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            if precision == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(x, edge_index)
            else:
                _ = model(x, edge_index)
    torch.cuda.synchronize()

    # Timed: 5 iterations
    num_iters = 5
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(num_iters):
            if precision == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(x, edge_index)
            else:
                _ = model(x, edge_index)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / num_iters
    nodes_per_sec = num_nodes / (avg_ms / 1000)

    params = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
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
            {"name": "nodes_per_sec", "value": nodes_per_sec},
        ],
    }))


if __name__ == '__main__':
    benchmark()
