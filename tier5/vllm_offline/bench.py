import sys
import time

def benchmark():
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vllm_offline: SKIPPED (vllm not installed)", file=sys.stderr)
        print("vllm_offline,128,0.000,0.00")
        return

    model_name = "facebook/opt-125m"
    max_tokens = 128

    prompts = [
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
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens)

    print(f"Loading model: {model_name}", file=sys.stderr)
    llm = LLM(model=model_name)

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
    tokens_per_sec = avg_tokens / (avg_ms / 1000)
    requests_per_sec = len(prompts) / (avg_ms / 1000)

    print(f"Average time: {avg_ms:.3f} ms", file=sys.stderr)
    print(f"Tokens/sec: {tokens_per_sec:.2f}", file=sys.stderr)
    print(f"Requests/sec: {requests_per_sec:.2f}", file=sys.stderr)

    print(f"vllm_offline,{max_tokens},{avg_ms:.3f},{tokens_per_sec:.2f}")

if __name__ == "__main__":
    benchmark()
