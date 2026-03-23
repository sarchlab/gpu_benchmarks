# Cara — Workspace Note

## Last Cycle
- Completed issue #166: Fixed mixbench default + vllm_offline duplicate param name
- Branch: ares/m21-milestone, commit 21995cc
- mixbench: Added 4.0 to flops_per_byte values (default was missing from values list)
- vllm_offline: Renamed non_scaling_param 'max_tokens' to 'max_output_tokens' (was duplicate of scaling_param name)

## Previous Cycles
- Completed issue #157: Added non_scaling_params to 14 Tier 2 batch A params.json files
- Completed issue #155: Fixed 7 params.json files to meet minimum value counts
- Completed issue #151: Created params.json for 13 Tier 5 benchmarks (batch A)
- Completed issue #136: Created tier2/altis_raytracing/ and tier2/chai_hsto/
