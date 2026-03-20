import json
import os
import sys
import time
import torch
import torchvision.models as models


def bench_variant(model, x, label, num_warmup, num_runs=5):
    """Run warmup + timed iterations, return avg ms."""
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            out = model(x)
    _ = out.sum().item()  # sync

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            out = model(x)
            _ = out.sum().item()  # force sync
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    ms = sum(times) / len(times)
    return ms


def benchmark():
    device = torch.device('mps')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '32'))
    image_size = int(os.environ.get('BENCH_PARAM_image_size', '224'))
    compile_mode = os.environ.get('BENCH_PARAM_compile_mode', 'default')
    num_warmup = int(os.environ.get('BENCH_PARAM_num_warmup', '5'))

    print(f"torch.compile benchmark (MPS) — batch_size={batch_size}, "
          f"image_size={image_size}, compile_mode={compile_mode}, "
          f"num_warmup={num_warmup}", file=sys.stderr)

    model = models.resnet50(weights=None).to(device)
    model.eval()

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    params = {
        "batch_size": batch_size,
        "image_size": image_size,
        "compile_mode": compile_mode,
        "num_warmup": num_warmup,
    }

    # Eager variant
    eager_ms = bench_variant(model, x, 'eager', num_warmup)
    eager_ips = batch_size / (eager_ms / 1000)
    print(f"  eager: {eager_ms:.3f} ms, {eager_ips:.2f} images/sec",
          file=sys.stderr)

    print(json.dumps({
        "type": "kernel",
        "name": "forward_eager",
        "time_ms": eager_ms,
        "params": params,
    }))

    # Compiled variant — torch.compile may not support MPS backend
    compiled_ms = None
    try:
        compiled_model = torch.compile(model, mode=compile_mode)
        compiled_ms = bench_variant(compiled_model, x, 'compiled', num_warmup)
        compiled_ips = batch_size / (compiled_ms / 1000)
        print(f"  compiled: {compiled_ms:.3f} ms, {compiled_ips:.2f} images/sec",
              file=sys.stderr)

        print(json.dumps({
            "type": "kernel",
            "name": "forward_compiled",
            "time_ms": compiled_ms,
            "params": params,
        }))
    except Exception as e:
        print(f'NOTE: torch.compile not supported on MPS backend, '
              f'skipping compiled variant ({e})', file=sys.stderr)
        compiled_ips = 0.0

    # Speedup
    if compiled_ms is not None and compiled_ms > 0:
        speedup = eager_ms / compiled_ms
        print(f"  speedup: {speedup:.2f}x", file=sys.stderr)
    else:
        speedup = 0.0
        print('  speedup: N/A (compiled variant skipped)', file=sys.stderr)

    metrics = [
        {"name": "eager_ms", "value": eager_ms},
        {"name": "eager_images_per_sec", "value": eager_ips},
    ]
    total_ms = eager_ms
    if compiled_ms is not None:
        metrics.append({"name": "compiled_ms", "value": compiled_ms})
        metrics.append({"name": "compiled_images_per_sec", "value": compiled_ips})
        metrics.append({"name": "speedup", "value": speedup})
        total_ms += compiled_ms

    print(json.dumps({
        "type": "summary",
        "total_time_ms": total_ms,
        "metrics": metrics,
    }))


if __name__ == '__main__':
    benchmark()
