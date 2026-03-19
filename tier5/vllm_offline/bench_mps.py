import sys

def benchmark():
    print("vllm_offline: SKIPPED (vLLM not supported on MPS)", file=sys.stderr)

if __name__ == "__main__":
    benchmark()
