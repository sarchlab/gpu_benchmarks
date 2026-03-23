# Apollo Verification — M22: CLI Tool + Tier 1 Benchmark Refactoring

## Cycle 2: Decision Made — FAIL

### Verification Results

1. **CLI tool (scripts/gpubench)**: PASS — all 5 commands work, schema correct, stdlib-only
2. **Tier 1 Group A (6 benchmarks, 12 files)**: PASS — all checks verified
3. **Tier 1 Group B (6 benchmarks, 12 files)**: 4/6 PASS, **2 FAIL**

### Bugs Found (confirmed by Apollo)

**BUG 1: atomic_operations (.hip + _metal.mm)**
- params.json values: `atomicAdd`, `atomicCAS`, `atomicMin`
- Code compares: `add`, `cas`, `min`
- Fix: update code to match params.json OR update params.json to match code

**BUG 2: warp_shuffle (.hip + _metal.mm)**  
- params.json values: `shfl_xor`, `shfl`, `shfl_up`, `shfl_down`
- Code compares: `xor`, `basic`, `up`, `down`
- Fix: update code to match params.json OR update params.json to match code

### Workers Used
- nora: CLI verification
- owen: Group A benchmark verification  
- pat: Group B benchmark verification
