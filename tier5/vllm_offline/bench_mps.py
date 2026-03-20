import json
import os
import sys


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

    print("vllm_offline: SKIPPED (vLLM not supported on MPS)", file=sys.stderr)

    print(json.dumps({"type": "kernel", "name": "generate", "time_ms": 0.0, "params": params}))
    print(json.dumps({"type": "summary", "total_time_ms": 0.0, "metrics": [
        {"name": "tokens_per_sec", "value": 0.0},
        {"name": "requests_per_sec", "value": 0.0},
    ]}))


if __name__ == "__main__":
    benchmark()
