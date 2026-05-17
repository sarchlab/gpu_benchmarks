import json
import os
import sys
import torch
import torchvision.models.detection as detection
import time

def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '4'))
    image_size = int(os.environ.get('BENCH_PARAM_image_size', '640'))
    num_classes = int(os.environ.get('BENCH_PARAM_num_classes', '91'))
    precision = os.environ.get('BENCH_PARAM_precision', 'fp32')
    num_warmup = 0

    print(f"Running retinanet_inference (MPS) with batch_size={batch_size}, image_size={image_size}, "
          f"num_classes={num_classes}, precision={precision}", file=sys.stderr)

    model = detection.retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes).to(device)
    model.eval()

    if precision == 'fp16':
        model = model.half()

    # RetinaNet in eval mode takes a list of tensors, not a batch tensor
    images = [torch.randn(3, image_size, image_size, device=device) for _ in range(batch_size)]
    if precision == 'fp16':
        images = [img.half() for img in images]

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if precision == 'amp':
                with torch.autocast('mps'):
                    out = model(images)
            else:
                out = model(images)
    torch.mps.synchronize()

    num_iters = 5

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            t0 = time.perf_counter()
            if precision == 'amp':
                with torch.autocast('mps'):
                    out = model(images)
            else:
                out = model(images)
            torch.mps.synchronize()
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
