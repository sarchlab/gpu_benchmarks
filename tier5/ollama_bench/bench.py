import json
import os
import sys
import time

try:
    import requests
except ImportError:
    # Still emit valid JSON for SKIPPED case
    params = {
        "num_predict": int(os.environ.get('BENCH_PARAM_num_predict', '128')),
        "model": os.environ.get('BENCH_PARAM_model', 'tinyllama'),
        "prompt_length": os.environ.get('BENCH_PARAM_prompt_length', 'medium'),
        "temperature": float(os.environ.get('BENCH_PARAM_temperature', '0.7')),
    }
    print(
        "ollama_bench: SKIPPED (requests not installed, run 'pip install requests')",
        file=sys.stderr,
    )
    print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
    print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [{"name": "tokens_per_sec", "value": 0.0}]}))
    sys.exit(0)

OLLAMA_URL = "http://localhost:11434/api/generate"
WARMUP = 0
ITERATIONS = 10

PROMPTS = {
    "short": "What is AI?",
    "medium": "Explain the theory of relativity in simple terms.",
    "long": "Write a detailed essay about the history of artificial intelligence, covering its origins in the 1950s, the AI winters, the resurgence of neural networks, and the modern era of large language models. Discuss the key milestones and breakthroughs along the way.",
}


def check_ollama():
    """Check if ollama is reachable."""
    try:
        resp = requests.get(
            "http://localhost:11434/api/tags", timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_available_model(preferred_model):
    """Get the first available model from ollama, preferring the specified model."""
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
        # Prefer requested model if available
        for m in models:
            if preferred_model.lower() in m.lower():
                return m
        # Otherwise use the first model
        return models[0]
    except Exception:
        return None


def run_inference(model, prompt, num_predict, temperature):
    """Run a single ollama inference and return (elapsed_ms, tokens_per_sec, eval_count)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
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
    # Read parameters from environment variables
    num_predict = int(os.environ.get('BENCH_PARAM_num_predict', '128'))
    model_name = os.environ.get('BENCH_PARAM_model', 'tinyllama')
    prompt_length = os.environ.get('BENCH_PARAM_prompt_length', 'medium')
    temperature = float(os.environ.get('BENCH_PARAM_temperature', '0.7'))
    global WARMUP
    WARMUP = 0

    params = {
        "num_predict": num_predict,
        "model": model_name,
        "prompt_length": prompt_length,
        "temperature": temperature,
    }

    prompt = PROMPTS.get(prompt_length, PROMPTS["medium"])

    if not check_ollama():
        print(
            "ollama_bench: SKIPPED (ollama not reachable at localhost:11434)",
            file=sys.stderr,
        )
        print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
        print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [{"name": "tokens_per_sec", "value": 0.0}]}))
        return

    model = get_available_model(model_name)
    if model is None:
        print(
            "ollama_bench: SKIPPED (no models available, run 'ollama pull tinyllama')",
            file=sys.stderr,
        )
        print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
        print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [{"name": "tokens_per_sec", "value": 0.0}]}))
        return

    print(f"Model: {model}", file=sys.stderr)
    print(f"Endpoint: {OLLAMA_URL}", file=sys.stderr)
    print(f"Tokens to generate: {num_predict}", file=sys.stderr)
    print(f"Prompt length: {prompt_length}, temperature: {temperature}", file=sys.stderr)

    # Warmup
    print(f"Warmup: {WARMUP} iterations", file=sys.stderr)
    for i in range(WARMUP):
        elapsed_ms, tps, eval_count = run_inference(model, prompt, num_predict, temperature)
        status = f"{tps:.2f} tok/s" if tps else "parse failed"
        print(f"  warmup {i+1}/{WARMUP}: {elapsed_ms:.1f} ms, {status}", file=sys.stderr)

    # Timed
    print(f"Benchmark: {ITERATIONS} iterations", file=sys.stderr)
    times = []
    tps_list = []
    for i in range(ITERATIONS):
        elapsed_ms, tps, eval_count = run_inference(model, prompt, num_predict, temperature)
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
