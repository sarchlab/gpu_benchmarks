import json
import os
import subprocess
import sys
import re
import time


def find_binary():
    """Find the llama-cli or main binary in the llama.cpp build directory."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama.cpp")
    candidates = [
        os.path.join(base, "build", "bin", "llama-cli"),
        os.path.join(base, "build", "bin", "main"),
        os.path.join(base, "llama-cli"),
        os.path.join(base, "main"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def find_model():
    """Find a GGUF model file in the models directory."""
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".gguf"):
                return os.path.join(models_dir, f)
    return None


def parse_tokens_per_sec(output):
    """Parse llama.cpp output for eval tokens/sec."""
    pattern = r"eval\s+time\s*=.*?(\d+\.\d+)\s*tokens per second"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    pattern2 = r"eval\s+.*?(\d+\.\d+)\s*tokens/s"
    match2 = re.search(pattern2, output)
    if match2:
        return float(match2.group(1))
    return None


def run_inference(binary, model, n_predict=128, n_gpu_layers=99, threads=4):
    """Run a single llama.cpp inference and return tokens/sec."""
    cmd = [
        binary,
        "-m", model,
        "-p", "The meaning of life is",
        "-n", str(n_predict),
        "-ngl", str(n_gpu_layers),
        "-t", str(threads),
        "--no-display-prompt",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + "\n" + result.stderr
    tps = parse_tokens_per_sec(combined)
    return tps


def benchmark():
    # Read parameters from environment variables
    n_predict = int(os.environ.get('BENCH_PARAM_n_predict', '128'))
    n_gpu_layers = int(os.environ.get('BENCH_PARAM_n_gpu_layers', '99'))
    quantization = os.environ.get('BENCH_PARAM_quantization', 'Q4_0')
    threads = int(os.environ.get('BENCH_PARAM_threads', '4'))
    num_warmup = 0

    params = {
        "n_predict": n_predict,
        "n_gpu_layers": n_gpu_layers,
        "quantization": quantization,
        "threads": threads,
    }

    binary = find_binary()
    if binary is None:
        print(
            "llamacpp_bench: SKIPPED (llama.cpp not built, run 'make setup')",
            file=sys.stderr,
        )
        print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
        print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [{"name": "tokens_per_sec", "value": 0.0}]}))
        return

    model = find_model()
    if model is None:
        print(
            "llamacpp_bench: SKIPPED (no GGUF model found, run 'make setup')",
            file=sys.stderr,
        )
        print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
        print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [{"name": "tokens_per_sec", "value": 0.0}]}))
        return

    print(f"Binary: {binary}", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"Tokens to generate: {n_predict}", file=sys.stderr)
    print(f"GPU layers: {n_gpu_layers}, quantization: {quantization}, threads: {threads}", file=sys.stderr)

    # Warmup
    print(f"Warmup: {num_warmup} iterations", file=sys.stderr)
    for i in range(num_warmup):
        tps = run_inference(binary, model, n_predict, n_gpu_layers, threads)
        status = f"{tps:.2f} tok/s" if tps else "parse failed"
        print(f"  warmup {i+1}/{num_warmup}: {status}", file=sys.stderr)

    # Timed
    print("Benchmark: 10 iterations", file=sys.stderr)
    times = []
    tps_list = []
    for i in range(10):
        t0 = time.perf_counter()
        tps = run_inference(binary, model, n_predict, n_gpu_layers, threads)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)
        if tps is not None:
            tps_list.append(tps)
        status = f"{tps:.2f} tok/s" if tps else "parse failed"
        print(
            f"  iter {i+1}/10: {elapsed_ms:.1f} ms, {status}",
            file=sys.stderr,
        )

    avg_ms = sum(times) / len(times)
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Average tokens/sec: {avg_tps:.2f}", file=sys.stderr)

    print(json.dumps({
        "type": "kernel",
        "name": "generate",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "tokens_per_sec", "value": avg_tps},
        ],
    }))


if __name__ == "__main__":
    benchmark()
