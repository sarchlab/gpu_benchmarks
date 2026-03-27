import json
import os
import sys
import torch
import torchvision.models as models
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '32'))
    image_size = int(os.environ.get('BENCH_PARAM_image_size', '224'))
    num_classes = int(os.environ.get('BENCH_PARAM_num_classes', '1000'))
    learning_rate = float(os.environ.get('BENCH_PARAM_learning_rate', '0.01'))
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    print(f"Running resnet50_train (MPS) with batch_size={batch_size}, image_size={image_size}, "
          f"num_classes={num_classes}, lr={learning_rate}", file=sys.stderr)

    model = models.resnet50(weights=None, num_classes=num_classes).to(device)
    model.train()

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    if num_warmup > 0:
        _ = loss.item()  # sync

    num_iters = 10

    # Timed
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        _ = loss.item()  # force sync
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    images_per_sec = batch_size / (avg_ms / 1000)

    params = {
        "batch_size": batch_size,
        "image_size": image_size,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "train_step",
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
