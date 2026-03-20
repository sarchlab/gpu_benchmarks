import json
import os
import sys
import time


SAMPLE_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where technology advances rapidly",
    "The most important scientific discovery of the century",
    "Once upon a time in a land far away there lived",
    "The key to understanding quantum computing lies in",
    "Climate change has become one of the most pressing",
    "The history of human civilization shows us that",
    "When we consider the implications of machine learning",
    "Space exploration has always been a dream of humanity",
    "The relationship between mathematics and physics is",
    "The evolution of programming languages has shaped",
    "In the realm of quantum mechanics particles behave",
    "The development of renewable energy sources is crucial",
    "Artificial neural networks are inspired by the structure",
    "The internet has fundamentally changed the way we",
    "Modern cryptography relies heavily on mathematical",
    "The study of genetics has revealed remarkable insights",
    "Robotics and automation are transforming industries",
    "The philosophy of mind explores questions about",
    "Data science combines statistics programming and domain",
    "The mysteries of dark matter and dark energy continue",
    "Natural language processing enables computers to understand",
    "The principles of thermodynamics govern the behavior",
    "Biotechnology has opened new frontiers in medicine",
    "The concept of parallel computing allows multiple",
    "Cybersecurity is becoming increasingly important as",
    "The theory of evolution by natural selection explains",
    "Virtual reality technology creates immersive digital",
    "The study of complex systems reveals emergent properties",
    "Blockchain technology provides a decentralized approach",
    "The field of materials science develops new substances",
    "Autonomous vehicles rely on a combination of sensors",
    "The mathematical foundations of computer science include",
    "Nanotechnology manipulates matter at the atomic scale",
    "The principles of game theory apply to economics",
    "Edge computing brings processing power closer to",
    "The human brain contains approximately one hundred billion",
    "Sustainable development balances economic growth with",
    "The standard model of particle physics describes",
    "Cloud computing has revolutionized how organizations",
    "The study of ecosystems reveals complex interactions",
    "Quantum entanglement is one of the most fascinating",
    "The development of antibiotics transformed modern",
    "Computer vision enables machines to interpret visual",
    "The principles of aerodynamics govern the flight of",
    "Distributed systems face challenges related to",
    "The human genome project mapped the entire sequence",
    "Machine learning algorithms can be categorized into",
    "The laws of conservation in physics state that",
    "Digital twins create virtual replicas of physical",
]


def benchmark():
    # Read parameters from environment variables
    max_tokens = int(os.environ.get('BENCH_PARAM_max_tokens', '128'))
    num_prompts = int(os.environ.get('BENCH_PARAM_num_prompts', '10'))
    model_name = os.environ.get('BENCH_PARAM_model_name', 'facebook/opt-125m')
    max_output_tokens = int(os.environ.get('BENCH_PARAM_max_output_tokens', '128'))

    params = {
        "max_tokens": max_tokens,
        "num_prompts": num_prompts,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
    }

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vllm_offline: SKIPPED (vllm not installed)", file=sys.stderr)
        print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
        print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [
            {"name": "tokens_per_sec", "value": 0.0},
            {"name": "requests_per_sec", "value": 0.0},
        ]}))
        return

    # Select prompts up to num_prompts (cycle if needed)
    prompts = []
    for i in range(num_prompts):
        prompts.append(SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)])

    sampling_params = SamplingParams(max_tokens=max_output_tokens)

    print(f"Loading model: {model_name}", file=sys.stderr)
    print(f"max_tokens={max_tokens}, num_prompts={num_prompts}, max_output_tokens={max_output_tokens}", file=sys.stderr)

    try:
        llm = LLM(model=model_name)
    except Exception as e:
        print(f"vllm_offline: SKIPPED (failed to load model: {e})", file=sys.stderr)
        print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
        print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [
            {"name": "tokens_per_sec", "value": 0.0},
            {"name": "requests_per_sec", "value": 0.0},
        ]}))
        return

    # Warmup
    print("Warmup: 5 iterations", file=sys.stderr)
    for i in range(5):
        _ = llm.generate(prompts, sampling_params)
        print(f"  warmup {i+1}/5 done", file=sys.stderr)

    # Timed
    print("Benchmark: 10 iterations", file=sys.stderr)
    times = []
    total_tokens_list = []
    for i in range(10):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_tokens_list.append(total_tokens)
        print(f"  iter {i+1}/10: {elapsed_ms:.1f} ms, {total_tokens} tokens generated", file=sys.stderr)

    avg_ms = sum(times) / len(times)
    avg_tokens = sum(total_tokens_list) / len(total_tokens_list)
    tokens_per_sec = avg_tokens / (avg_ms / 1000) if avg_ms > 0 else 0.0
    requests_per_sec = len(prompts) / (avg_ms / 1000) if avg_ms > 0 else 0.0

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Tokens/sec: {tokens_per_sec:.2f}", file=sys.stderr)
    print(f"Requests/sec: {requests_per_sec:.2f}", file=sys.stderr)

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
            {"name": "tokens_per_sec", "value": tokens_per_sec},
            {"name": "requests_per_sec", "value": requests_per_sec},
        ],
    }))


if __name__ == "__main__":
    benchmark()
