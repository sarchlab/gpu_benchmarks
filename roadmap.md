# GPU Benchmarks — Project Roadmap

## Project Goal
Create an extensive GPGPU benchmark suite covering 5 tiers:
- **Tier 1**: Microbenchmarks (memory bandwidth, latency, compute throughput, etc.)
- **Tier 2**: Application benchmarks (SHOC, PolyBench, Rodinia, Parboil, etc.)
- **Tier 3**: Library benchmarks (cuBLAS, cuDNN, CUTLASS, etc.)
- **Tier 4**: Alternative language benchmarks (Triton, OpenCL, etc.)
- **Tier 5**: ML workloads (inference, training, typical deep learning models)

## Success Criteria
Deliver a comprehensive, well-organized benchmark suite that researchers and engineers can use to evaluate GPGPU hardware and software stacks.

---

## Milestones

### M1 — Research & Report (cycles budget: 3, started cycle 1)
**Goal:** Produce a comprehensive research report identifying the key benchmarks to include in each tier, justifying selections, and proposing a concrete implementation plan.

**Acceptance:** A `benchmark_report.md` file committed to the repo containing:
- For each tier: a curated list of benchmarks with descriptions, sources, and rationale
- Implementation complexity assessment for each benchmark
- Recommended priority/ordering for implementation

**Status:** 🔄 In Progress

---

### M2 — Tier 1 Implementation: Microbenchmarks (planned)
Implement the selected Tier 1 microbenchmarks (memory bandwidth, latency, occupancy, etc.)

### M3 — Tier 2 Implementation: Application Benchmarks (planned)
Implement/integrate selected Tier 2 benchmarks from SHOC, Rodinia, PolyBench, Parboil.

### M4 — Tier 3 Implementation: Library Benchmarks (planned)
Implement/integrate cuBLAS, CUTLASS, and other library benchmarks.

### M5 — Tier 4 & 5 Implementation: Triton + ML Workloads (planned)
Implement Triton-based benchmarks and representative ML training/inference workloads.

### M6 — Integration & Harness (planned)
Unified benchmark harness: run-all script, result collection, reporting infrastructure.

---

## Lessons Learned
- (none yet — project just started)

## Cycle History
| Cycle | Phase | Agent | Action |
|-------|-------|-------|--------|
| 1 | Planning | Athena | Created roadmap, hired researcher |
