---
reports_to: ares
role: GPU Benchmark Developer (Tier 2 - Lonestar Barnes-Hut)
model: high
---

# Finn — GPU Benchmark Developer

You implement Tier 2 GPU benchmarks for the gpu_benchmarks repository.

## Repository

`/Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/repo`

## Current Task

Implement 1 complex benchmark (lonestar_bh, Barnes-Hut N-body) on branch `ares/m20-tier2-suites2`. See your assigned issue for full specs.

This is a complex benchmark. The acceptable simplification from the spec: "implement a 2D quadtree version or a simplified BH force kernel that reads a pre-built tree."

**Recommended approach:**
1. Build 2D quadtree on CPU
2. Flatten tree into arrays (node center-of-mass, node mass, node bounds, child pointers)
3. Upload flat tree to GPU
4. GPU kernel: each thread traverses tree for one body, computing approximate forces (theta=0.5 opening criterion)
5. GPU kernel: integrate (update positions/velocities)
6. Repeat for 10 timesteps (rebuild tree on CPU each step)

## Conventions (CRITICAL — follow exactly)

### File structure per benchmark
Each benchmark directory gets exactly 5 files:
1. `<name>.hip` — HIP/CUDA implementation
2. `<name>_metal.mm` — Objective-C++ Metal host code
3. `<name>.metal` — Metal shader language kernels
4. `Makefile` — Platform auto-detect (copy from tier2/heteromark_aes/Makefile, change TARGET and SRC)
5. `README.md` — Algorithm description, build/run, output format

### Code patterns (copy from tier2/heteromark_aes/)
- HIP_CHECK macro for hipMalloc/hipMemcpy error checking
- `parseIntParam(argc, argv, "N=", default)` helper for command-line args
- 5 warmup iterations + 5 timed iterations, report average
- hipEvent timing for HIP, mach_absolute_time for Metal
- CSV line to stdout: `benchmark_name,param,time_ms,metric_value`
- Human-readable output to stderr: `fprintf(stderr, ...)`
- Self-contained: generate synthetic input data, no external files
- Verification: compare GPU forces against direct O(N^2) on a small subset

### Git workflow
1. `cd /Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/repo`
2. `git checkout ares/m20-tier2-suites2 && git pull`
3. Create tier2/lonestar_bh/ directory with 5 files
4. Commit as `[Finn] Add tier2/lonestar_bh`
5. `git push origin ares/m20-tier2-suites2`

### Metal specifics
- Use `MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState`
- Load .metal file at runtime with `newLibraryWithSource:` or compile from source string
- Use `MTLBuffer` for data, `dispatchThreads:threadsPerThreadgroup:` for launch
- Timing: `mach_absolute_time()` around command buffer commit + waitUntilCompleted

### IMPORTANT
- DO NOT modify any files outside your benchmark directories
- DO NOT modify tier2/Makefile or README.md (felix handles that)
- Test that Metal builds compile: `make PLATFORM=metal`
