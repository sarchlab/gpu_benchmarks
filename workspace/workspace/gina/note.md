# Gina — Workspace Note

## Last completed
- Issue #114: Implemented `tier5/moe_inference` benchmark
- Commit: 17d09bd on `ares/m18-advanced-ml`
- Files: bench.py, bench_mps.py, Makefile, README.md
- Verified MPS version runs (warmup + timed runs produce valid output)
- CUDA bench.py syntax verified (no CUDA GPU to test runtime)
- All model components defined inline (Expert, TopKGating, MoELayer, TransformerBlock, MoETransformer)
- Config: hidden=512, experts=4, top_k=2, layers=4, heads=8, vocab=50257, batch=4, seq_len=256, 64 new tokens

## Patterns used
- Inline nn.Module classes (no HuggingFace, no downloads)
- model.eval() + @torch.no_grad() for inference
- Autoregressive generation: 256-token prompt → 64 new tokens
- 5 warmup + 10 timed runs, per-run timing
- CUDA: torch.cuda.Event timing; MPS: time.perf_counter + .cpu() sync
- Human-readable output to stderr, CSV to stdout
