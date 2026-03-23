# Ares Workspace Notes

## Current: Fix round for M22 verification (3 cycles remaining)

### Apollo's Feedback
Two enum mismatch bugs in 4 files:
1. atomic_operations: code uses "add"/"cas"/"min" but params.json says "atomicAdd"/"atomicCAS"/"atomicMin"
2. warp_shuffle: code uses "xor"/"basic"/"up"/"down" but params.json says "shfl_xor"/"shfl"/"shfl_up"/"shfl_down"

### Fix Plan
- Issue #174: Leo fixes atomic_operations (2 files)
- Issue #175: Max fixes warp_shuffle (2 files)

### Branch
`ares/m22-cli-benchmarks` (main working branch)

Both fixes are simple string replacements in strcmp calls. Should be done in one cycle.
