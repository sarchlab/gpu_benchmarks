import json
import sys
import time

try:
    import requests
except ImportError:
    print(
        "ollama_bench: SKIPPED (requests not installed, run 'pip install requests')",
        file=sys.stderr,
    )
    print("ollama_bench,128,0.000,0.00")
    sys.exit(0)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"
PROMPT = "Explain the theory of relativity in simple terms."
N_PREDICT = 128
WARMUP = 5
ITERATIONS = 10


def check_ollama():
    """Check if ollama is reachable."""
    try:
        resp = requests.get(
            "http://localhost:11434/api/tags", timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_available_model():
    """Get the first available model from ollama, preferring tinyllama."""
    try:
        resp = requests.get(
            "http://localhost:11434/api/tags", timeout=5
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            return None
        # Prefer tinyllama if available
        for m in models:
            if "tinyllama" in m.lower():
                return m
        # Otherwise use the first model
        return models[0]
    except Exception:
        return None


def run_inference(model, prompt, n_predict):
    """Run a single ollama inference and return (elapsed_ms, tokens_per_sec, eval_count)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": n_predict,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000

    if resp.status_code != 200:
        return elapsed_ms, None, None

    data = resp.json()

    # Ollama returns timing info in the response
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)

    if eval_duration_ns > 0 and eval_count > 0:
        eval_duration_s = eval_duration_ns / 1e9
        tokens_per_sec = eval_count / eval_duration_s
    else:
        # Fallback: estimate from wall-clock time
        tokens_per_sec = eval_count / (elapsed_ms / 1000) if elapsed_ms > 0 and eval_count > 0 else 0.0

    return elapsed_ms, tokens_per_sec, eval_count


def benchmark():
    if not check_ollama():
        print(
            "ollama_bench: SKIPPED (ollama not reachable at localhost:11434)",
            file=sys.stderr,
        )
        print("ollama_bench,128,0.000,0.00")
        return

    model = get_available_model()
    if model is None:
        print(
            "ollama_bench: SKIPPED (no models available, run 'ollama pull tinyllama')",
            file=sys.stderr,
        )
        print("ollama_bench,128,0.000,0.00")
        return

    print(f"Model: {model}", file=sys.stderr)
    print(f"Endpoint: {OLLAMA_URL}", file=sys.stderr)
    print(f"Tokens to generate: {N_PREDICT}", file=sys.stderr)

    # Warmup
    print(f"Warmup: {WARMUP} iterations", file=sys.stderr)
    for i in range(WARMUP):
        elapsed_ms, tps, eval_count = run_inference(model, PROMPT, N_PREDICT)
        status = f"{tps:.2f} tok/s" if tps else "parse failed"
        print(f"  warmup {i+1}/{WARMUP}: {elapsed_ms:.1f} ms, {status}", file=sys.stderr)

    # Timed
    print(f"Benchmark: {ITERATIONS} iterations", file=sys.stderr)
    times = []
    tps_list = []
    for i in range(ITERATIONS):
        elapsed_ms, tps, eval_count = run_inference(model, PROMPT, N_PREDICT)
        times.append(elapsed_ms)
        if tps is not None:
            tps_list.append(tps)
        status = f"{tps:.2f} tok/s" if tps else "parse failed"
        print(
            f"  iter {i+1}/{ITERATIONS}: {elapsed_ms:.1f} ms, {status}",
            file=sys.stderr,
        )

    avg_ms = sum(times) / len(times)
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Average tokens/sec: {avg_tps:.2f}", file=sys.stderr)

    print(f"ollama_bench,{N_PREDICT},{avg_ms:.3f},{avg_tps:.2f}")


if __name__ == "__main__":
    benchmark()
