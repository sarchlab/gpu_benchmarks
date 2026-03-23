# Ben - Workspace Note

## Last completed: Issue #165 - Fix uppercase param names in 7 tier4 params.json files

Fixed 20 uppercase parameter names across 7 tier4 params.json files. Commit: 545133e on branch `ares/m21-milestone`.

### Changes (name field only, lowercase):
- **tier4/babelstream**: N → n
- **tier4/triton_dropout**: N → n, BLOCK_SIZE → block_size
- **tier4/triton_flash_attention**: S → s, B → b, H → h, D → d, BLOCK_M → block_m, BLOCK_N → block_n
- **tier4/triton_gemm**: M → m, BLOCK_M → block_m, BLOCK_N → block_n, BLOCK_K → block_k, GROUP_SIZE_M → group_size_m
- **tier4/triton_layer_norm**: N → n, M → m
- **tier4/triton_softmax**: N → n, M → m
- **tier4/triton_vector_add**: N → n, BLOCK_SIZE → block_size

### Previous: Issue #161 - Fix 8 params.json validation errors
Fixed 8 params.json files. Commit: 12b54a1.

### Previous: Issue #156 - Fix params.json for Tier 1 + Tier 3
Fixed 6 params.json files.

### Previous: Issue #154 - Fix 22 Tier 2 params.json scaling values
Fixed all 22 Tier 2 params.json files.
