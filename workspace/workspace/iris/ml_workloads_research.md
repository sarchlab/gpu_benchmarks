# Tier 5 Expansion: Modern ML/LLM Workload Benchmarks Research

**Author:** Iris  
**Date:** Current cycle  
**Issue:** #103  
**Scope:** Define 18 specific ML/LLM benchmarks for M16, M17, M18 milestones

---

## Executive Summary

The current Tier 5 has 8 benchmarks focused on classic ML workloads (ResNet-50, BERT, nanoGPT, ViT, SDPA, torch.compile). These cover PyTorch-native CNN/Transformer training and inference with synthetic data.

The expansion adds **18 new benchmarks** in 3 batches of 6, covering:
- **M16:** LLM inference frameworks (vLLM, llama.cpp, ollama) + popular OSS models (LLaMA, Qwen, DeepSeek)
- **M17:** More LLM models (GLM, Phi, Mistral) + PyTorch training workloads (BERT fine-tuning, GPT-NeoX training, Whisper inference)
- **M18:** Diffusion models (Stable Diffusion, SDXL) + advanced ML patterns (MoE, RAG, CoT, multi-agent)

---

## Current Tier 5 Inventory (8 benchmarks)

| Benchmark | Type | Model | Metric | Deps |
|-----------|------|-------|--------|------|
| `resnet50_train` | Training | ResNet-50 (torchvision) | images/sec | torch, torchvision |
| `resnet50_infer` | Inference | ResNet-50 (torchvision) | images/sec | torch, torchvision |
| `bert_inference` | Inference | BERT-base (HuggingFace config) | sequences/sec | torch, transformers |
| `nanogpt_train` | Training | GPT-2 (inline, 6L/6H/384E) | tokens/sec | torch |
| `vit_inference` | Inference | ViT-B/16 (torchvision) | images/sec | torch, torchvision |
| `sdpa_bench` | Microbench | F.scaled_dot_product_attention | TFLOPS | torch |
| `gpt2_generate` | Inference | GPT-2 (inline, 6L/6H/384E) | tokens/sec | torch |
| `torch_compile_bench` | Inference | ResNet-50 eager vs compiled | images/sec | torch, torchvision |

**Observations:**
- All use synthetic/random data (no model weight downloads)
- All have `bench.py` (CUDA) + `bench_mps.py` (Apple MPS) pattern
- CSV output: `benchmark_name,size,time_ms,metric_value`
- Models are either inline-defined (nanoGPT) or from torchvision/transformers (no weight downloads, `weights=None`)
- All are single-GPU, single-node benchmarks

---

## M16 Batch: LLM Inference Frameworks + Popular Models (6 benchmarks)

These benchmarks test the dominant LLM inference stack: inference frameworks (vLLM, llama.cpp, ollama) and the most popular open-source LLM models (LLaMA, Qwen, DeepSeek).

---

### M16-1: `tier5/vllm_bench/`

**What it measures:** LLM inference throughput and latency using vLLM's PagedAttention engine. Measures tokens/sec for both prefill and decode phases.

**Metric:** tokens/sec (total output tokens / wall time), time-to-first-token (TTFT) in ms

**Python dependencies:** `pip install vllm torch`

**Model weights:** Requires download. Uses `facebook/opt-125m` (small, ~250MB, ungated) for testing; notes `meta-llama/Llama-2-7b-hf` as full variant.

**Model size/config:**
- Small variant (testing): OPT-125M (125M params, ~250MB)
- Full variant (production): LLaMA-2-7B (7B params, ~14GB FP16)

**Implementation approach:** Script-only pattern. `bench.py` uses vLLM's `LLM` class for offline batch inference benchmarking. No `bench_mps.py` — vLLM is Linux+NVIDIA/AMD only. Apple Metal: print skip message with explanatory note.

```
bench.py:
  from vllm import LLM, SamplingParams
  model = LLM(model="facebook/opt-125m", ...)
  # Warmup: 2 requests
  # Timed: 10 batches of 8 prompts, 128 output tokens each
  # Report: tokens/sec, TTFT
bench_mps.py:
  print("vllm_bench,skip,0,0  # vLLM requires Linux+NVIDIA/AMD GPU")
```

**CSV output:** `vllm_bench,<batch>x<output_len>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Medium (vLLM install is heavy but API is simple)

**Platform notes:**
- ⚠️ vLLM does NOT support macOS/MPS. Linux only with NVIDIA CUDA or AMD ROCm.
- `bench_mps.py` should output a skip line (the CSV format should still be valid but with 0 values)
- Makefile needs: `setup: pip3 install vllm` (Linux only)
- Requires ~4GB GPU memory for OPT-125M

---

### M16-2: `tier5/llamacpp_bench/`

**What it measures:** LLM inference throughput using llama.cpp (C/C++ GGML-based inference). Tests quantized model inference speed (GGUF format).

**Metric:** tokens/sec (generation speed)

**Python dependencies:** None for core (C/C++ build). Optional: `pip install huggingface-hub` for model download helper.

**Model weights:** Requires download. Uses a small GGUF model: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (Q4_K_M quantized, ~670MB). Notes `TheBloke/Llama-2-7B-Chat-GGUF` as full variant.

**Model size/config:**
- Small variant (testing): TinyLlama-1.1B Q4_K_M (~670MB GGUF)
- Full variant (production): LLaMA-2-7B-Chat Q4_K_M (~4.1GB GGUF)

**Implementation approach:** Makefile wrapper around llama.cpp binary. The benchmark directory contains:
- `Makefile` that clones/builds llama.cpp (or uses pre-built), downloads model GGUF, and runs `llama-bench` (llama.cpp's built-in benchmark tool)
- `bench.sh` — shell script that wraps llama.cpp's `llama-bench` binary, parses output, and emits CSV
- No `bench.py` / `bench_mps.py` — this is a C/C++ benchmark

```
bench.sh:
  # Run llama.cpp's built-in benchmark with -ngl (GPU layers)
  ./llama.cpp/build/bin/llama-bench \
    -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p 512 -n 128 -ngl 99 -t 4
  # Parse output: extract tokens/sec
  # Emit CSV: llamacpp_bench,<model>,<time_ms>,<tokens_per_sec>
```

**CSV output:** `llamacpp_bench,<model_name>,<avg_ms>,<tokens_per_sec>`

**Complexity:** High (requires building llama.cpp from source, downloading GGUF model, different build system)

**Platform notes:**
- ✅ Supports CUDA (cuBLAS), ROCm (hipBLAS), Metal (via Metal backend built-in to llama.cpp)
- llama.cpp has first-class Apple Metal support via `LLAMA_METAL=ON`
- Makefile needs: `setup` target that `git clone`s llama.cpp, `cmake -B build -DLLAMA_CUDA=ON` (or METAL/HIP), and downloads the GGUF model
- `clean` target removes `llama.cpp/` and `models/` directories

---

### M16-3: `tier5/ollama_bench/`

**What it measures:** LLM inference throughput via Ollama's REST API. Tests the end-to-end serving stack including model loading, tokenization, and generation.

**Metric:** tokens/sec (generation throughput), time-to-first-token (TTFT) in ms

**Python dependencies:** `pip install requests`

**Model weights:** Ollama manages models internally. Uses `tinyllama:1.1b` (small, ~637MB) for testing. Notes `llama2:7b` as full variant.

**Model size/config:**
- Small variant (testing): tinyllama:1.1b (~637MB)
- Full variant (production): llama2:7b (~3.8GB quantized)

**Implementation approach:** Makefile wrapper with `curl`/Python script calling Ollama's REST API. Assumes Ollama is installed and running.

```
bench.py:
  import requests, time, json
  # Pre-pull model: requests.post("http://localhost:11434/api/pull", json={"name": "tinyllama:1.1b"})
  # Warmup: 2 generation requests
  # Timed: 10 requests, each generating 128 tokens
  # Parse streaming JSON response for eval_count, eval_duration
  # Report: tokens/sec = eval_count / (eval_duration / 1e9)
bench_mps.py:
  # Same as bench.py — Ollama runs the same on macOS
```

**CSV output:** `ollama_bench,<model>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Medium (requires Ollama installed as a system service; API is simple)

**Platform notes:**
- ✅ Ollama supports macOS (Metal), Linux (CUDA), and Linux (ROCm)
- The `bench_mps.py` is identical to `bench.py` since Ollama handles GPU selection internally
- Makefile `setup` target: check if `ollama` binary exists, if not print install instructions
- `setup` also runs `ollama pull tinyllama:1.1b`
- Must handle case where Ollama server is not running (auto-start or error message)

---

### M16-4: `tier5/llama_inference/`

**What it measures:** LLaMA model family inference using HuggingFace transformers. Tests autoregressive generation speed with the smallest LLaMA-architecture model available.

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights (no download needed). Creates a `LlamaForCausalLM` from a minimal `LlamaConfig`. Notes real model `meta-llama/Llama-3.2-1B` (gated, needs HF token) as full variant.

**Model size/config:**
- Small variant (testing): Custom LlamaConfig — vocab_size=32000, hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=1376, max_position_embeddings=512. ~30M params. Random weights.
- Full variant (production): LLaMA-3.2-1B (1.2B params, ~2.4GB FP16, requires HF token for gated access)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  from transformers import LlamaForCausalLM, LlamaConfig
  config = LlamaConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=4,
                       num_attention_heads=8, intermediate_size=1376, max_position_embeddings=512)
  model = LlamaForCausalLM(config).to('cuda').eval()
  # Random prompt: 32 tokens
  # Generate 128 new tokens (model.generate with max_new_tokens=128)
  # 3 warmup + 5 timed runs
  # Report: tokens/sec = 128 / (avg_ms / 1000)
```

**CSV output:** `llama_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (HuggingFace API, random weights, no download)

**Platform notes:**
- ✅ Works on CUDA, ROCm (via torch), and MPS (Apple)
- Uses `transformers.LlamaForCausalLM` which is available in transformers >= 4.31
- Random weights mean results reflect architecture overhead, not real model quality
- For real-world results, user can swap config for a real model checkpoint

---

### M16-5: `tier5/qwen_inference/`

**What it measures:** Qwen model family inference using HuggingFace transformers. Tests the Qwen2 architecture (popular Chinese LLM family from Alibaba).

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights. Creates a `Qwen2ForCausalLM` from a minimal `Qwen2Config`. Notes `Qwen/Qwen2.5-0.5B` (ungated, ~1GB) as full variant.

**Model size/config:**
- Small variant (testing): Custom Qwen2Config — vocab_size=32000, hidden_size=512, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4 (GQA), intermediate_size=1376, max_position_embeddings=512. ~30M params. Random weights.
- Full variant (production): Qwen2.5-0.5B (0.5B params, ~1GB FP16, ungated on HuggingFace)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  from transformers import Qwen2ForCausalLM, Qwen2Config
  config = Qwen2Config(vocab_size=32000, hidden_size=512, num_hidden_layers=4,
                       num_attention_heads=8, num_key_value_heads=4,
                       intermediate_size=1376, max_position_embeddings=512)
  model = Qwen2ForCausalLM(config).to('cuda').eval()
  # Same benchmark loop as llama_inference
  # 32-token prompt, generate 128 tokens, 3 warmup + 5 timed
```

**CSV output:** `qwen_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (same pattern as llama_inference, just different model class)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- Qwen2 uses Grouped Query Attention (GQA) with `num_key_value_heads < num_attention_heads`
- Requires transformers >= 4.37 for Qwen2 support
- Ungated model — no HF token needed for full variant download

---

### M16-6: `tier5/deepseek_inference/`

**What it measures:** DeepSeek model family inference. Tests the DeepSeek architecture (popular Chinese LLM from DeepSeek AI, known for strong coding/math performance).

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights. Creates a model using `LlamaForCausalLM` with DeepSeek-like config (DeepSeek v1 uses LLaMA architecture). Notes `deepseek-ai/deepseek-llm-7b-base` (ungated, ~14GB FP16) as full variant.

**Model size/config:**
- Small variant (testing): LlamaConfig with DeepSeek-like params — vocab_size=32256, hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=1376, max_position_embeddings=512, rms_norm_eps=1e-6. ~30M params. Random weights.
- Full variant (production): deepseek-llm-7b-base (7B params, ~14GB FP16, ungated)
- Note: DeepSeek-V2/V3 uses MLA (Multi-head Latent Attention) and MoE, which is more complex — covered in M18 as a separate MoE benchmark.

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  from transformers import LlamaForCausalLM, LlamaConfig
  # DeepSeek v1 is LLaMA-based
  config = LlamaConfig(vocab_size=32256, hidden_size=512, num_hidden_layers=4,
                       num_attention_heads=8, intermediate_size=1376,
                       max_position_embeddings=512, rms_norm_eps=1e-6)
  model = LlamaForCausalLM(config).to('cuda').eval()
  # Same benchmark loop: 32-token prompt, generate 128 tokens
```

**CSV output:** `deepseek_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (LLaMA-compatible architecture, same pattern as llama_inference)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- DeepSeek v1 uses standard LLaMA architecture, so `LlamaForCausalLM` works directly
- DeepSeek-V2/V3 has specialized architecture (MLA + MoE) — deferred to M18's MoE benchmark
- Ungated model — no HF token needed

---

## M17 Batch: More LLM Models + Training Workloads (6 benchmarks)

This batch adds more LLM model diversity (GLM, Phi, Mistral) and critical training/inference workloads (BERT fine-tuning, GPT-NeoX training, Whisper).

---

### M17-1: `tier5/glm_inference/`

**What it measures:** GLM (General Language Model) inference, specifically ChatGLM architecture from Tsinghua/Zhipu AI. Tests the GLM architecture which uses a unique prefix-LM structure.

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights with custom config. Notes `THUDM/chatglm3-6b` (ungated, ~12GB FP16) as full variant.

**Model size/config:**
- Small variant (testing): Custom architecture — since ChatGLM3 uses a custom model class, we use a comparable transformer: `AutoModelForCausalLM` with a manually-defined config that mirrors GLM's architecture. hidden_size=512, num_layers=4, num_attention_heads=8, ffn_hidden_size=1376, vocab_size=65024 (GLM's tokenizer size), multi_query_attention=True. ~35M params. Random weights.
- Full variant (production): ChatGLM3-6B (6.2B params, ~12GB FP16, ungated)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Since ChatGLM uses a custom architecture registered in transformers, use `AutoModelForCausalLM` with trust_remote_code for the full model. For the synthetic small variant, define a simple transformer inline (similar to nanogpt but with GLM's multi-query attention pattern).

```
bench.py:
  import torch, torch.nn as nn, math
  # Define a GLM-style model inline with:
  # - Multi-query attention (1 KV head, 8 query heads)
  # - RoPE positional encoding
  # - SwiGLU activation (like ChatGLM3)
  # 4 layers, hidden=512, heads=8, kv_heads=1, ffn=1376
  # ~35M params
  model = GLM(config).to('cuda').eval()
  # 32-token prompt, generate 128 tokens, 3 warmup + 5 timed
```

**CSV output:** `glm_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Medium (GLM has unique architecture features — multi-query attention, SwiGLU; needs inline model definition)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- ChatGLM3's trust_remote_code requirement complicates HF integration — inline model is safer for benchmarking
- Full model uses `sentencepiece` tokenizer (additional dep for full variant)

---

### M17-2: `tier5/phi_inference/`

**What it measures:** Microsoft Phi model family inference. Tests the Phi-2/Phi-3 architecture, known for high efficiency at small model sizes.

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights. Creates `PhiForCausalLM` from `PhiConfig`. Notes `microsoft/phi-2` (ungated, ~5.4GB FP16) as full variant.

**Model size/config:**
- Small variant (testing): Custom PhiConfig — vocab_size=51200, hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048, max_position_embeddings=512, partial_rotary_factor=0.5. ~30M params. Random weights.
- Full variant (production): Phi-2 (2.7B params, ~5.4GB FP16, ungated on HuggingFace)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  from transformers import PhiForCausalLM, PhiConfig
  config = PhiConfig(vocab_size=51200, hidden_size=512, num_hidden_layers=4,
                     num_attention_heads=8, intermediate_size=2048,
                     max_position_embeddings=512, partial_rotary_factor=0.5)
  model = PhiForCausalLM(config).to('cuda').eval()
  # 32-token prompt, generate 128 tokens, 3 warmup + 5 timed
```

**CSV output:** `phi_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (standard HuggingFace pattern, PhiConfig available in transformers >= 4.36)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- Phi uses partial rotary embeddings (unlike full RoPE in LLaMA) — tests a different attention variant
- Ungated model — no HF token needed

---

### M17-3: `tier5/mistral_inference/`

**What it measures:** Mistral model inference. Tests the Mistral architecture, known for sliding window attention and GQA.

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights. Creates `MistralForCausalLM` from `MistralConfig`. Notes `mistralai/Mistral-7B-v0.1` (ungated, ~14GB FP16) as full variant.

**Model size/config:**
- Small variant (testing): Custom MistralConfig — vocab_size=32000, hidden_size=512, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4 (GQA), intermediate_size=1376, max_position_embeddings=512, sliding_window=256. ~30M params. Random weights.
- Full variant (production): Mistral-7B-v0.1 (7.3B params, ~14GB FP16, ungated)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  from transformers import MistralForCausalLM, MistralConfig
  config = MistralConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=4,
                         num_attention_heads=8, num_key_value_heads=4,
                         intermediate_size=1376, max_position_embeddings=512,
                         sliding_window=256)
  model = MistralForCausalLM(config).to('cuda').eval()
  # 32-token prompt, generate 128 tokens, 3 warmup + 5 timed
```

**CSV output:** `mistral_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (standard HuggingFace pattern)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- Mistral's sliding window attention tests a different attention pattern than full attention
- GQA with 4 KV heads (same as Qwen2 benchmark, different architecture)
- Ungated model — no HF token needed

---

### M17-4: `tier5/bert_finetune/`

**What it measures:** BERT fine-tuning (training) throughput. Measures forward + backward + optimizer step speed on a sequence classification task.

**Metric:** sequences/sec (training throughput)

**Python dependencies:** `pip install torch transformers`

**Model weights:** No download needed. Uses `BertForSequenceClassification` with random weights from `BertConfig`.

**Model size/config:**
- Small variant (testing): BERT-base — vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, num_labels=2. ~110M params. Random weights.
- Full variant: Same architecture, real pre-trained weights `bert-base-uncased` (~440MB)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Similar structure to `resnet50_train`.

```
bench.py:
  from transformers import BertForSequenceClassification, BertConfig
  config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                      num_attention_heads=12, intermediate_size=3072, num_labels=2)
  model = BertForSequenceClassification(config).to('cuda')
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
  # batch=16, seq_len=128, random input_ids + random labels
  # 5 warmup + 10 timed forward+backward+optimizer.step()
  # Report: sequences/sec = batch * 10 / total_time_s
```

**CSV output:** `bert_finetune,<batch_size>,<avg_ms>,<sequences_per_sec>`

**Complexity:** Low (follows existing resnet50_train pattern)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- Complements existing `bert_inference` — now we have both train and infer for BERT
- Uses AdamW optimizer (standard for BERT fine-tuning)

---

### M17-5: `tier5/gptneox_train/`

**What it measures:** GPT-NeoX style training throughput. Tests training a modern decoder-only transformer with rotary embeddings (RoPE) and parallel attention+MLP computation.

**Metric:** tokens/sec (training throughput)

**Python dependencies:** `pip install torch transformers`

**Model weights:** No download needed. Uses `GPTNeoXForCausalLM` with random weights from `GPTNeoXConfig`.

**Model size/config:**
- Small variant (testing): Custom GPTNeoXConfig — vocab_size=50432, hidden_size=512, num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048, max_position_embeddings=512, rotary_pct=0.25. ~40M params. Random weights.
- Full variant: GPT-NeoX-20B (20B params) or Pythia-1.4B (1.4B params, ~2.8GB FP16, ungated)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Similar to `nanogpt_train` but uses HuggingFace model class.

```
bench.py:
  from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
  config = GPTNeoXConfig(vocab_size=50432, hidden_size=512, num_hidden_layers=6,
                         num_attention_heads=8, intermediate_size=2048,
                         max_position_embeddings=512, rotary_pct=0.25)
  model = GPTNeoXForCausalLM(config).to('cuda')
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  # batch=4, seq_len=512, random token IDs
  # 5 warmup + 10 timed forward+backward+optimizer.step()
  # Report: tokens/sec = batch * seq_len * 10 / total_time_s
```

**CSV output:** `gptneox_train,<tokens_per_step>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (standard HuggingFace training pattern)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- GPT-NeoX uses rotary embeddings (RoPE) — tests a different positional encoding than nanoGPT (learned PE)
- Parallel attention+MLP is a distinctive architectural feature
- Complements `nanogpt_train` (inline model) with a HuggingFace-based training benchmark

---

### M17-6: `tier5/whisper_inference/`

**What it measures:** OpenAI Whisper speech-to-text inference throughput. Tests encoder-decoder transformer architecture on audio processing workload.

**Metric:** audio_seconds/wall_second (real-time factor), or seconds of audio processed per second

**Python dependencies:** `pip install torch transformers`

**Model weights:** No download needed. Uses `WhisperForConditionalGeneration` with random weights from `WhisperConfig`.

**Model size/config:**
- Small variant (testing): Custom WhisperConfig — d_model=256, encoder_layers=4, decoder_layers=4, encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=1024, decoder_ffn_dim=1024, vocab_size=51865, num_mel_bins=80, max_source_positions=1500, max_target_positions=448. ~20M params. Random weights + random mel-spectrogram input.
- Full variant: whisper-small (244M params, ~500MB) or whisper-base (74M params, ~150MB)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Generates random mel-spectrogram features and runs encoder+decoder.

```
bench.py:
  from transformers import WhisperForConditionalGeneration, WhisperConfig
  config = WhisperConfig(d_model=256, encoder_layers=4, decoder_layers=4,
                         encoder_attention_heads=4, decoder_attention_heads=4,
                         encoder_ffn_dim=1024, decoder_ffn_dim=1024,
                         vocab_size=51865, num_mel_bins=80,
                         max_source_positions=1500, max_target_positions=448)
  model = WhisperForConditionalGeneration(config).to('cuda').eval()
  # Random mel features: [batch=4, n_mels=80, n_frames=3000] (30 sec of audio)
  # model.generate(input_features, max_new_tokens=224)
  # 3 warmup + 5 timed runs
  # Report: audio_seconds_per_wall_second = (batch * 30) / wall_time_s
```

**CSV output:** `whisper_inference,<audio_secs>,<avg_ms>,<realtime_factor>`

**Complexity:** Medium (encoder-decoder architecture is more complex than decoder-only; generate() behavior differs)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- First encoder-decoder model in the suite (all others are decoder-only or encoder-only)
- Tests a non-text modality (audio → text) — important for multimodal GPU evaluation
- Random mel-spectrogram input means output tokens are meaningless, but throughput is still valid

---

## M18 Batch: Diffusion Models + Advanced ML Patterns (6 benchmarks)

This batch covers the most advanced and rapidly-evolving ML workloads: diffusion image generation, Mixture-of-Experts, RAG, Chain-of-Thought, and multi-agent execution.

---

### M18-1: `tier5/stable_diffusion_inference/`

**What it measures:** Stable Diffusion image generation inference throughput. Tests the UNet denoising diffusion pipeline with text conditioning.

**Metric:** images/sec (generation throughput), seconds/image

**Python dependencies:** `pip install torch diffusers transformers accelerate`

**Model weights:** Uses random weights for UNet, VAE, and text encoder. No download needed. Notes `runwayml/stable-diffusion-v1-5` (ungated, ~4GB) as full variant.

**Model size/config:**
- Small variant (testing): Minimal UNet2DConditionModel config — in_channels=4, out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), block_out_channels=(128, 256), cross_attention_dim=512, sample_size=32. With tiny VAE and tiny CLIP text encoder. ~15M total params. Image size: 256×256 with 20 diffusion steps.
- Full variant: SD v1.5 (860M UNet + 123M VAE + 123M text encoder, 512×512, 50 steps)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Build a minimal diffusion pipeline manually (not using `StableDiffusionPipeline` which requires real weights).

```
bench.py:
  from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
  from transformers import CLIPTextModel, CLIPTextConfig
  # Build tiny pipeline components with random weights
  unet = UNet2DConditionModel(...minimal config...).to('cuda')
  scheduler = DDPMScheduler(num_train_timesteps=1000)
  # Run 20-step denoising loop
  # 3 warmup + 5 timed runs
  # Report: images/sec = batch / (avg_ms / 1000)
```

**CSV output:** `stable_diffusion_inference,<steps>x<resolution>,<avg_ms>,<images_per_sec>`

**Complexity:** Medium (multiple model components; diffusion loop is more complex than simple forward pass)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- `diffusers` library handles the diffusion pipeline logic
- With random weights, generated images are noise — but throughput measurement is valid
- Tests a fundamentally different workload pattern than LLMs (iterative denoising vs autoregressive generation)

---

### M18-2: `tier5/sdxl_inference/`

**What it measures:** Stable Diffusion XL (SDXL) inference throughput. Tests the larger SDXL architecture with two text encoders and higher-resolution output.

**Metric:** images/sec (generation throughput)

**Python dependencies:** `pip install torch diffusers transformers accelerate`

**Model weights:** Uses random weights. Notes `stabilityai/stable-diffusion-xl-base-1.0` (ungated, ~6.5GB) as full variant.

**Model size/config:**
- Small variant (testing): Minimal SDXL-like UNet with dual cross-attention (two text encoder outputs). UNet config: block_out_channels=(128, 256), cross_attention_dim=512, addition_embed_type="text_time". Sample size: 64 (maps to 512×512 at 8x VAE). 20 diffusion steps. ~20M total params.
- Full variant: SDXL-Base-1.0 (2.6B UNet, 1024×1024, 50 steps)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Similar to `stable_diffusion_inference` but with SDXL's dual text encoder conditioning.

```
bench.py:
  from diffusers import UNet2DConditionModel, DDPMScheduler
  # Build SDXL-style UNet with addition_embed_type="text_time"
  # Two text encoder outputs concatenated
  # 20-step denoising loop
  # 3 warmup + 5 timed runs
```

**CSV output:** `sdxl_inference,<steps>x<resolution>,<avg_ms>,<images_per_sec>`

**Complexity:** Medium (SDXL's dual encoder and time-text embedding adds complexity over SD v1)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- SDXL is significantly more compute-heavy than SD v1.5 even at small scale
- Tests the "next-gen" diffusion architecture that's becoming the production standard

---

### M18-3: `tier5/moe_inference/`

**What it measures:** Mixture-of-Experts (MoE) model inference throughput. Tests sparse expert routing and load balancing, the key architectural innovation in models like Mixtral and DeepSeek-V2/V3.

**Metric:** tokens/sec (generation throughput)

**Python dependencies:** `pip install torch transformers accelerate`

**Model weights:** Uses random weights. Creates `MixtralForCausalLM` from `MixtralConfig` (Mixtral is the reference MoE implementation in HuggingFace). Notes `mistralai/Mixtral-8x7B-v0.1` (ungated, ~93GB FP16) as full variant.

**Model size/config:**
- Small variant (testing): Custom MixtralConfig — vocab_size=32000, hidden_size=512, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4 (GQA), intermediate_size=1376, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=512. ~80M total params (4 experts × ~15M each + shared ~20M). Random weights.
- Full variant: Mixtral-8x7B-v0.1 (46.7B total params, 8 experts with top-2 routing, ~93GB FP16)

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  from transformers import MixtralForCausalLM, MixtralConfig
  config = MixtralConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=4,
                         num_attention_heads=8, num_key_value_heads=4,
                         intermediate_size=1376, num_local_experts=4,
                         num_experts_per_tok=2, max_position_embeddings=512)
  model = MixtralForCausalLM(config).to('cuda').eval()
  # 32-token prompt, generate 128 tokens, 3 warmup + 5 timed
  # Report: tokens/sec = 128 / (avg_ms / 1000)
```

**CSV output:** `moe_inference,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Medium (MoE routing adds complexity; MixtralConfig is available in transformers >= 4.36)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- MoE models have irregular compute patterns (different experts activated per token) — tests GPU's ability to handle dynamic workloads
- With only 4 experts and 512 hidden dim, the small variant fits easily in any GPU
- This tests the MoE pattern generally — relevant to Mixtral, DeepSeek-V2/V3, Switch Transformer, etc.

---

### M18-4: `tier5/rag_pipeline/`

**What it measures:** Retrieval-Augmented Generation (RAG) pipeline throughput. Benchmarks the full RAG loop: encode query → retrieve from vector store → generate with context.

**Metric:** queries/sec (end-to-end RAG throughput)

**Python dependencies:** `pip install torch transformers faiss-cpu` (or `faiss-gpu` for GPU vector search)

**Model weights:** No download needed. Uses random weights for both the retrieval encoder and the generator. Synthetic document embeddings in FAISS index.

**Model size/config:**
- Embedding model: BertModel with hidden_size=256, num_hidden_layers=2. ~10M params.
- Generator: GPT-2 style (same as nanogpt_train inline model, 4 layers). ~20M params.
- Vector store: FAISS IndexFlatL2 with 10,000 synthetic document embeddings (dim=256).
- Retrieval: top-5 documents per query, concatenated as context for generation.

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. All components built inline.

```
bench.py:
  import torch, faiss
  from transformers import BertModel, BertConfig
  # 1. Build embedding model (small BERT encoder)
  encoder = BertModel(BertConfig(hidden_size=256, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=512,
                                  vocab_size=30522)).to('cuda').eval()
  # 2. Build FAISS index with 10K random embeddings
  index = faiss.IndexFlatL2(256)
  index.add(np.random.randn(10000, 256).astype('float32'))
  # 3. Build generator (inline GPT, 4 layers, 256 hidden)
  generator = GPT(vocab_size=30522, n_layer=4, n_head=4, n_embd=256, block_size=512).to('cuda').eval()
  # 4. RAG loop: encode query → FAISS search → generate with context
  # 3 warmup + 10 timed runs
  # Report: queries/sec
```

**CSV output:** `rag_pipeline,<top_k>,<avg_ms>,<queries_per_sec>`

**Complexity:** High (multiple components, FAISS integration, context window management)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS (FAISS CPU for retrieval, GPU for models)
- `faiss-cpu` works everywhere; `faiss-gpu` only on NVIDIA CUDA
- Measures the full RAG pipeline, not just generation — includes encoding + retrieval + generation
- FAISS search is CPU-bound (or GPU with faiss-gpu); models are GPU-bound
- This is a system-level benchmark, not a single-model benchmark

---

### M18-5: `tier5/cot_reasoning/`

**What it measures:** Chain-of-Thought (CoT) reasoning throughput. Benchmarks multi-step autoregressive generation where the model generates a long reasoning chain before producing a final answer. Tests sustained generation over long sequences.

**Metric:** tokens/sec (sustained generation throughput over long sequences)

**Python dependencies:** `pip install torch`

**Model weights:** No download needed. Uses inline GPT-2 model (same architecture as nanogpt_train/gpt2_generate).

**Model size/config:**
- Model: GPT-2 inline model — n_layer=6, n_head=6, n_embd=384, block_size=1024 (larger context than gpt2_generate's 256). ~25M params. Random weights.
- CoT simulation: Generate 512 tokens (simulating "thinking" steps) then 64 more tokens (simulating "answer"). Total generation: 576 tokens from a 32-token prompt.

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern. Reuses the GPT model from `gpt2_generate` but with a much longer generation length and larger context window.

```
bench.py:
  # Same inline GPT model as gpt2_generate, but with block_size=1024
  model = GPT(vocab_size=50257, n_layer=6, n_head=6, n_embd=384, block_size=1024).to('cuda').eval()
  prompt = torch.randint(0, 50257, (1, 32), device='cuda')
  # Generate 576 tokens (simulating long CoT reasoning)
  # Key difference from gpt2_generate: much longer generation stresses
  #   KV-cache (if implemented) and attention over long sequences
  # 2 warmup + 5 timed runs
  # Report: tokens/sec = 576 / (avg_ms / 1000)
```

**CSV output:** `cot_reasoning,<num_tokens>,<avg_ms>,<tokens_per_sec>`

**Complexity:** Low (reuses existing inline GPT model, just larger context and longer generation)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- The key stress test is sustained generation with growing context — attention cost grows quadratically with sequence length
- Without KV-cache, generation of 576 tokens with full recomputation is very expensive — measures worst-case CoT performance
- Compare tokens/sec with `gpt2_generate` (128 tokens, block_size=256) to see the effect of longer sequences

---

### M18-6: `tier5/multi_agent_bench/`

**What it measures:** Multi-agent LLM execution throughput. Simulates multiple independent LLM agents running concurrently on the same GPU, measuring aggregate throughput and per-agent latency.

**Metric:** total_tokens/sec (aggregate throughput across all agents), per-agent tokens/sec

**Python dependencies:** `pip install torch`

**Model weights:** No download needed. Uses inline GPT-2 model (shared weights across agents).

**Model size/config:**
- Model: GPT-2 inline model — n_layer=6, n_head=6, n_embd=384, block_size=256. ~25M params. Random weights. Single model instance shared by all agents.
- Agents: 4 concurrent agents, each with its own prompt (batch dimension), generating independently.
- Approach: Batch agents together as a single batch of 4 — measures GPU utilization when handling multiple concurrent requests (similar to continuous batching in inference servers).

**Implementation approach:** Standard `bench.py` + `bench_mps.py` pattern.

```
bench.py:
  # Same inline GPT model as gpt2_generate
  model = GPT(vocab_size=50257, n_layer=6, n_head=6, n_embd=384, block_size=256).to('cuda').eval()

  # Simulate 4 agents with different prompts (batched)
  num_agents = 4
  prompt_lens = [32, 48, 24, 40]  # Different prompt lengths per agent
  max_prompt = max(prompt_lens)
  # Pad to same length, use attention masking
  prompts = torch.randint(0, 50257, (num_agents, max_prompt), device='cuda')

  # Generate 128 tokens for all agents simultaneously
  # This tests GPU's ability to handle multiple concurrent generation tasks
  # 3 warmup + 5 timed runs
  # Report: total_tokens/sec = num_agents * 128 / (avg_ms / 1000)
  #         per_agent_tokens/sec = 128 / (avg_ms / 1000)
```

**CSV output:** Two lines:
```
multi_agent_bench,4_agents_total,<avg_ms>,<total_tokens_per_sec>
multi_agent_bench,per_agent,<avg_ms>,<per_agent_tokens_per_sec>
```

**Complexity:** Low-Medium (batched generation is straightforward; the key insight is measuring multi-agent overhead vs single-agent throughput)

**Platform notes:**
- ✅ Works on CUDA, ROCm, and MPS
- Compare total throughput vs `gpt2_generate` × 4 to measure batching efficiency
- In production, multi-agent systems like AutoGen, CrewAI, etc. run multiple LLM calls concurrently — this benchmark captures that workload pattern
- Variable prompt lengths test padding/masking overhead

---

## Summary Table

### M16 — LLM Inference Frameworks + Popular Models

| # | Directory | What it Tests | Metric | Model Weights | Complexity | MPS Support |
|---|-----------|--------------|--------|---------------|------------|-------------|
| 1 | `tier5/vllm_bench/` | vLLM serving engine | tokens/sec | Download (OPT-125M) | Medium | ❌ Skip |
| 2 | `tier5/llamacpp_bench/` | llama.cpp C/C++ inference | tokens/sec | Download (GGUF) | High | ✅ Metal |
| 3 | `tier5/ollama_bench/` | Ollama REST API serving | tokens/sec | Ollama-managed | Medium | ✅ |
| 4 | `tier5/llama_inference/` | LLaMA architecture (HF) | tokens/sec | Random weights | Low | ✅ |
| 5 | `tier5/qwen_inference/` | Qwen2 architecture (HF) | tokens/sec | Random weights | Low | ✅ |
| 6 | `tier5/deepseek_inference/` | DeepSeek v1 architecture (HF) | tokens/sec | Random weights | Low | ✅ |

### M17 — More LLM Models + Training Workloads

| # | Directory | What it Tests | Metric | Model Weights | Complexity | MPS Support |
|---|-----------|--------------|--------|---------------|------------|-------------|
| 1 | `tier5/glm_inference/` | ChatGLM architecture | tokens/sec | Random weights | Medium | ✅ |
| 2 | `tier5/phi_inference/` | Microsoft Phi architecture (HF) | tokens/sec | Random weights | Low | ✅ |
| 3 | `tier5/mistral_inference/` | Mistral (sliding window attn, HF) | tokens/sec | Random weights | Low | ✅ |
| 4 | `tier5/bert_finetune/` | BERT training (fine-tuning) | sequences/sec | Random weights | Low | ✅ |
| 5 | `tier5/gptneox_train/` | GPT-NeoX training (RoPE) | tokens/sec | Random weights | Low | ✅ |
| 6 | `tier5/whisper_inference/` | Whisper speech-to-text inference | realtime_factor | Random weights | Medium | ✅ |

### M18 — Diffusion + Advanced Patterns

| # | Directory | What it Tests | Metric | Model Weights | Complexity | MPS Support |
|---|-----------|--------------|--------|---------------|------------|-------------|
| 1 | `tier5/stable_diffusion_inference/` | SD v1 diffusion pipeline | images/sec | Random weights | Medium | ✅ |
| 2 | `tier5/sdxl_inference/` | SDXL dual-encoder diffusion | images/sec | Random weights | Medium | ✅ |
| 3 | `tier5/moe_inference/` | Mixtral MoE routing | tokens/sec | Random weights | Medium | ✅ |
| 4 | `tier5/rag_pipeline/` | Full RAG pipeline | queries/sec | Random + FAISS | High | ✅ |
| 5 | `tier5/cot_reasoning/` | Long CoT generation (576 tokens) | tokens/sec | Random weights | Low | ✅ |
| 6 | `tier5/multi_agent_bench/` | Multi-agent concurrent LLM | total_tokens/sec | Random weights | Low-Medium | ✅ |

---

## Dependency Summary

### New pip packages needed (cumulative per milestone)

**M16:**
- `vllm` (Linux only, heavy — includes torch, triton, etc.)
- `requests` (for ollama)
- `huggingface-hub` (for model downloads)
- `transformers`, `accelerate` (already used in bert_inference)

**M17:**
- `transformers`, `accelerate` (same as M16)
- No new dependencies beyond existing

**M18:**
- `diffusers` (for Stable Diffusion / SDXL)
- `faiss-cpu` (for RAG pipeline vector search)
- No new dependencies for cot_reasoning and multi_agent_bench (pure torch)

### Build system notes

| Benchmark | Build System | Special Requirements |
|-----------|-------------|---------------------|
| `vllm_bench` | Python only | vLLM pip install (Linux only, no MPS) |
| `llamacpp_bench` | Makefile → cmake (llama.cpp) + shell | Clones llama.cpp repo, builds from source, downloads GGUF |
| `ollama_bench` | Python + system binary | Ollama must be installed and running as service |
| All others | Python only (standard pattern) | Standard `bench.py` + `bench_mps.py` pattern |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| vLLM install fails (heavy deps, GPU-specific) | M16-1 blocked | Fall back to skip message; document install requirements |
| llama.cpp build complexity (cmake, multiple backends) | M16-2 complex | Provide detailed Makefile with fallback to CPU-only build |
| Ollama not installed | M16-3 blocked | Makefile checks for `ollama` binary; prints install instructions if missing |
| HuggingFace transformers version too old | M16-4,5,6, M17 blocked | Document minimum version (>=4.37); add version check in bench.py |
| ChatGLM custom model class issues | M17-1 more complex | Use inline model definition instead of trust_remote_code |
| FAISS GPU not available (RAG) | M18-4 slower | Fall back to faiss-cpu; CPU retrieval is fast for 10K docs |
| Long generation in CoT (576 tokens) may be slow without KV-cache | M18-5 slow | Acceptable — the point is measuring sustained generation performance |

---

## Implementation Priority Within Each Milestone

### M16 (recommended order):
1. `llama_inference` (Low complexity, establishes HF LLM pattern)
2. `qwen_inference` (Low, same pattern)
3. `deepseek_inference` (Low, same pattern)
4. `vllm_bench` (Medium, Linux-only constraint)
5. `ollama_bench` (Medium, external dependency)
6. `llamacpp_bench` (High, C/C++ build system)

### M17 (recommended order):
1. `phi_inference` (Low, standard HF pattern)
2. `mistral_inference` (Low, standard HF pattern)
3. `bert_finetune` (Low, follows resnet50_train pattern)
4. `gptneox_train` (Low, HF training pattern)
5. `whisper_inference` (Medium, encoder-decoder)
6. `glm_inference` (Medium, custom architecture)

### M18 (recommended order):
1. `cot_reasoning` (Low, reuses inline GPT)
2. `multi_agent_bench` (Low-Medium, reuses inline GPT)
3. `moe_inference` (Medium, MixtralConfig in HF)
4. `stable_diffusion_inference` (Medium, diffusers library)
5. `sdxl_inference` (Medium, builds on SD)
6. `rag_pipeline` (High, multiple components)

---

## Appendix: Architecture Comparison of Proposed LLM Benchmarks

| Model | Attention | Positional Encoding | FFN | KV Heads | Special |
|-------|-----------|-------------------|-----|----------|---------|
| LLaMA | Full MHA → GQA | RoPE | SwiGLU | = or < Q heads | Industry standard |
| Qwen2 | GQA | RoPE | SwiGLU | 4 (< 8 Q heads) | Chinese LLM, strong multilingual |
| DeepSeek v1 | Full MHA | RoPE | SiLU | = Q heads | LLaMA-compatible |
| ChatGLM3 | MQA (1 KV head) | RoPE | SwiGLU | 1 | Unique prefix-LM design |
| Phi-2 | Full MHA | Partial RoPE | GELU | = Q heads | Efficient small models |
| Mistral | GQA + Sliding Window | RoPE | SiLU | 4 (< 8 Q heads) | Sliding window attention |
| GPT-NeoX | Full MHA | RoPE (partial) | GELU | = Q heads | Parallel attn+MLP |
| Mixtral (MoE) | GQA | RoPE | SiLU + Router | 4 (< 8 Q heads) | 4-of-8 expert routing |

This diversity ensures the benchmark suite tests different GPU computation patterns:
- **Full attention vs sliding window** (Mistral)
- **MHA vs GQA vs MQA** (varying KV-cache sizes)
- **Dense FFN vs MoE routing** (Mixtral)
- **Different activation functions** (SwiGLU, GELU, SiLU)
- **Encoder-decoder vs decoder-only** (Whisper vs all LLMs)
- **Iterative denoising vs autoregressive** (Stable Diffusion vs LLMs)
