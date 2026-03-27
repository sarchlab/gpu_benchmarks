import json
import os
import sys
import torch
import torchvision.models as models


def bench_variant(model, x, label, num_warmup, num_runs=5):
    """Run warmup + timed iterations, return avg ms."""
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    ms = total_ms / num_runs
    return ms


def benchmark():
    device = torch.device('cuda')

    # Read parameters from environment variables
    batch_size = int(os.environ.get('BENCH_PARAM_batch_size', '32'))
    image_size = int(os.environ.get('BENCH_PARAM_image_size', '224'))
    compile_mode = os.environ.get('BENCH_PARAM_compile_mode', 'default')
    num_warmup = int(os.environ.get('BENCH_PARAM_warmup', '0'))

    print(f"torch.compile benchmark — batch_size={batch_size}, "
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

    # Compiled variant
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

    # Speedup
    speedup = eager_ms / compiled_ms if compiled_ms > 0 else 0.0
    print(f"  speedup: {speedup:.2f}x", file=sys.stderr)

    print(json.dumps({
        "type": "summary",
        "total_time_ms": eager_ms + compiled_ms,
        "metrics": [
            {"name": "eager_ms", "value": eager_ms},
            {"name": "compiled_ms", "value": compiled_ms},
            {"name": "eager_images_per_sec", "value": eager_ips},
            {"name": "compiled_images_per_sec", "value": compiled_ips},
            {"name": "speedup", "value": speedup},
        ],
    }))


if __name__ == '__main__':
    benchmark()
