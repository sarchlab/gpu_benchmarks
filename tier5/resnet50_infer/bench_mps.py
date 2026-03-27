import json
import os
import sys
import torch
import torchvision.models as models
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '64'))
    image_size = int(os.environ.get('BENCH_PARAM_image_size', '224'))
    num_classes = int(os.environ.get('BENCH_PARAM_num_classes', '1000'))
    precision = os.environ.get('BENCH_PARAM_precision', 'fp32')
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    print(f"Running resnet50_infer (MPS) with batch_size={batch_size}, image_size={image_size}, "
          f"num_classes={num_classes}, precision={precision}", file=sys.stderr)

    model = models.resnet50(weights=None, num_classes=num_classes).to(device)
    model.eval()

    if precision == 'fp16':
        model = model.half()

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if precision == 'fp16':
        x = x.half()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if precision == 'amp':
                with torch.autocast('mps'):
                    out = model(x)
            else:
                out = model(x)
    if num_warmup > 0:
        _ = out.sum().item()  # sync

    num_iters = 10

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            t0 = time.perf_counter()
            if precision == 'amp':
                with torch.autocast('mps'):
                    out = model(x)
            else:
                out = model(x)
            _ = out.sum().item()  # force sync
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    images_per_sec = batch_size / (avg_ms / 1000)

    params = {
        "batch_size": batch_size,
        "image_size": image_size,
        "num_classes": num_classes,
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
            {"name": "images_per_sec", "value": images_per_sec},
        ],
    }))

if __name__ == '__main__':
    benchmark()
