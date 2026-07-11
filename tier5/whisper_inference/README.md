# Whisper Inference Benchmark

PyTorch benchmark: OpenAI Whisper audio model inference (WhisperForConditionalGeneration),
d_model=256, encoder_layers=4, decoder_layers=4, encoder_attention_heads=4,
decoder_attention_heads=4, vocab_size=51865.
Input: random mel-spectrogram tensor (batch=4, 80 mel bins, 3000 time frames).
Generates 64 tokens (simulating transcription).
Reports tokens/sec.

Model instantiated from WhisperConfig with random weights — no external model downloads required.

## Requirements

- Python 3.8+
- PyTorch 2.x
- HuggingFace Transformers
- CUDA GPU (NVIDIA) or ROCm GPU (AMD)

## Build & Run

### CUDA (NVIDIA)
```
make run PLATFORM=cuda
```

### ROCm (AMD)
```
make run PLATFORM=rocm
```

## Output Format

CSV: `benchmark_name,batch,time_ms,tokens_per_sec`

Example:
```
whisper_inference,4,456.789,560.44
```

## Metric Formula

tokens/sec = (64 * batch) / (avg_time_ms / 1000)
