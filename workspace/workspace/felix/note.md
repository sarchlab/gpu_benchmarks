# Felix — Workspace Notes

## Last Completed (Issue #160)
- Fixed 10 params.json files in Tier 4 + Tier 5 on `ares/m21-milestone`
- Tier 4 (3 files): Added num_times to babelstream, num_warps to triton_softmax and triton_vector_add
- Tier 5 (6 files added): threads to llamacpp_bench, temperature to ollama_bench, precision to resnet50_infer, num_warmup to torch_compile_bench, precision to vit_inference, max_tokens to vllm_offline
- Tier 5 (1 file trimmed): Removed batch_size from moe_inference (6→5 non_scaling_params)
- Commit: 9f3cff7, pushed to origin
- All JSON validated, all param counts within 3-5 range
