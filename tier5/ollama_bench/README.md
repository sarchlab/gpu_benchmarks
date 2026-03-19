# Ollama Inference Benchmark

HTTP API-based inference benchmark for [Ollama](https://ollama.com/).
Sends generation requests to a locally running Ollama server and measures
token generation throughput.

## Requirements

- Ollama installed and running (`ollama serve`)
- A model pulled (e.g., `ollama pull tinyllama`)
- Python 3.8+
- `requests` library (`pip install requests`)

## Setup

```bash
make setup
```

This will print installation instructions and install the `requests` Python package.

You must manually:
1. Install ollama (see [ollama.com](https://ollama.com/))
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull tinyllama`

## Run

```bash
make run
```

The benchmark auto-detects the first available model (preferring tinyllama).

### Platform Override

```bash
make run PLATFORM=cuda    # Linux NVIDIA
make run PLATFORM=rocm    # Linux AMD
make run PLATFORM=metal   # macOS Apple Silicon
```

Since Ollama handles GPU backend selection internally, all platforms use
the same HTTP API. The `bench_mps.py` variant is functionally identical
to `bench.py`.

## Output Format

CSV to stdout: `benchmark_name,size,time_ms,metric`

Example:
```
ollama_bench,128,2345.678,54.32
```

- **size**: number of tokens requested (num_predict)
- **time_ms**: average wall-clock time per inference call
- **metric**: tokens/sec as reported by Ollama's eval timing

## Metric

Tokens per second from Ollama's `eval_duration` and `eval_count` response
fields. Falls back to wall-clock estimation if those fields are unavailable.
