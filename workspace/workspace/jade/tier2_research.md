# Tier 2 Benchmark Suite Research Report

**Author:** Jade  
**Date:** 2026-03-19  
**Issue:** #102 — Identify 8+ new Tier 2 benchmark suites

---

## Executive Summary

This report identifies **10 new benchmark suites** for Tier 2 expansion, with 2–4 representative benchmarks recommended from each. The current Tier 2 has 33 benchmarks from 5 suites (SHOC, Rodinia, PolyBench, Parboil, CUDA Samples). Adding these suites would bring coverage to **15 suites** and **~65+ total benchmarks**, significantly diversifying workload coverage across graph analytics, scientific computing, ML primitives, and heterogeneous computing patterns.

### Selection Criteria

Each suite was evaluated on:
- **Self-contained feasibility**: Can benchmarks generate their own input data (no large external datasets)?
- **Multi-platform portability**: Can kernels be ported to HIP + Metal?
- **Implementation complexity**: Can we implement with Make-only build?
- **Workload diversity**: Does it cover compute patterns not already in Tier 2?

---

## Currently Implemented Suites (for reference)

| Suite | Count | Key Patterns |
|-------|-------|-------------|
| SHOC | 9 | Memory, FFT, GEMM, reduction, scan, sort, SpMV, stencil, triad |
| Rodinia | 9 | BFS, backprop, Gaussian, hotspot, k-means, LUD, NW, pathfinder, SRAD |
| PolyBench | 9 | 2D conv, 3MM, ATAX, BiCG, FDTD-2D, GEMM, Gram-Schmidt, Jacobi-2D, MVT |
| Parboil | 4 | Histogram, SGEMM, SpMV, stencil |
| CUDA Samples | 1 | Transpose |

---

## Suite 1: Lonestar GPU

**Source:** https://github.com/IntelligentSoftwareSystems/Galois (Lonestar GPU component)  
**Origin:** UT Austin Intelligent Software Systems group  
**Description:** GPU implementations of irregular graph algorithms with worklist-based parallelism. Focuses on graph analytics workloads with irregular memory access patterns and dynamic parallelism — patterns poorly covered by existing suites.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `lonestar_sssp` | Single-Source Shortest Path (Bellman-Ford/near-far) | Medium | CSR graph, iterative relaxation. Synthetic graph generation straightforward. |
| `lonestar_bh` | Barnes-Hut N-body simulation | High | Tree-based force computation. Very different from typical GPU kernels. |
| `lonestar_dmr` | Delaunay Mesh Refinement | High | Irregular mesh operations, worklist-driven. |
| `lonestar_mst` | Minimum Spanning Tree (Borůvka's) | Medium | Graph algorithm with edge contraction. |

### Portability Assessment
- **HIP**: Straightforward — original code is CUDA, direct HIP translation.
- **Metal**: Medium difficulty — no dynamic parallelism on Metal, but worklist patterns translate to compute shaders with indirect dispatch.
- **Self-contained**: Yes — CSR graphs can be generated synthetically (similar to existing `rodinia_bfs`).

### Priority: **HIGH** — Fills the "irregular graph algorithms" gap.

**Recommended picks:** `lonestar_sssp`, `lonestar_bh` (2 benchmarks)

---

## Suite 2: Altis Benchmark Suite

**Source:** https://github.com/utcs-scea/altis  
**Origin:** UT Austin / ORNL Future Technologies Group  
**Description:** A modernized CUDA benchmark suite designed as a successor to Rodinia. Organized into 3 difficulty levels. Benchmarks are well-documented, have standardized command-line interfaces, and are designed for architectural studies. Many overlap with Rodinia/Parboil but with improved implementations.

### Recommended Benchmarks (non-overlapping with existing suites)

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `altis_cfd` | Computational Fluid Dynamics (Euler equations) | Medium | Unstructured mesh solver, irregular memory patterns. |
| `altis_particlefilter` | Bayesian particle filter | Medium | Monte Carlo estimation with random number generation. |
| `altis_raytracing` | GPU ray tracing (simple scene) | Medium | Ray-sphere/triangle intersections, divergent control flow. |
| `altis_where` | Stream compaction (prefix-sum based) | Low | Fundamental GPU primitive, simple kernel. |

### Portability Assessment
- **HIP**: Direct — Altis already provides some HIP ports.
- **Metal**: Low–Medium — kernels are compute-shader compatible.
- **Self-contained**: Yes — all use synthetic or procedurally generated data.

### Priority: **HIGH** — Well-maintained, modern implementations, fills CFD/ray-tracing gaps.

**Recommended picks:** `altis_cfd`, `altis_particlefilter`, `altis_raytracing` (3 benchmarks)

---

## Suite 3: HeteroMark

**Source:** https://github.com/NUCAR-DEV/Hetero-Mark  
**Origin:** Northeastern University NUCAR group  
**Description:** Benchmarks specifically designed for heterogeneous computing characterization. Targets CPU-GPU interaction patterns, data movement, and task scheduling. Each benchmark is designed to stress a different aspect of heterogeneous execution.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `heteromark_aes` | AES-256 encryption | Low | Regular data-parallel, byte-level operations. Tests integer/bitwise throughput. |
| `heteromark_fir` | FIR filter (convolution) | Low | 1D convolution with configurable taps. Simple, self-contained. |
| `heteromark_kmeans2` | K-means (different impl from Rodinia) | Low | Alternative k-means with different parallelization strategy. |
| `heteromark_pagerank` | PageRank graph algorithm | Medium | Iterative sparse matrix-vector multiply pattern. |

### Portability Assessment
- **HIP**: Direct — HeteroMark already has OpenCL/CUDA/HIP versions.
- **Metal**: Low difficulty — all kernels are standard compute patterns.
- **Self-contained**: Yes — all generate synthetic input.

### Priority: **HIGH** — Easy to implement, good heterogeneous computing coverage.

**Recommended picks:** `heteromark_aes`, `heteromark_fir`, `heteromark_pagerank` (3 benchmarks)

---

## Suite 4: NAS Parallel Benchmarks — GPU (NPB-GPU)

**Source:** https://github.com/GMAP/NPB-GPU (CUDA ports) and various academic implementations  
**Origin:** NASA Advanced Supercomputing Division; GPU ports by multiple groups  
**Description:** The NAS Parallel Benchmarks are the gold standard for HPC performance evaluation. GPU ports exist for several kernels. These benchmarks test large-scale parallel patterns: conjugate gradient, multigrid, FFT-based PDEs, and integer sorting.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `npb_cg` | Conjugate Gradient (sparse linear system) | Medium | Irregular memory access, SpMV-heavy. Class S/W sizes are self-contained. |
| `npb_ep` | Embarrassingly Parallel (random number gen) | Low | Gaussian random pairs via Marsaglia polar method. Pure compute, no communication. |
| `npb_ft` | 3D FFT-based PDE solver | High | Multi-dimensional FFT, tests memory bandwidth + compute. |
| `npb_is` | Integer Sort (bucket sort) | Medium | Large-scale parallel histogram/sort. Tests atomic operations. |

### Portability Assessment
- **HIP**: Medium — CUDA ports exist, hipify is direct.
- **Metal**: Medium — EP and IS are straightforward. FT requires custom FFT (no cuFFT). CG is compute-shader friendly.
- **Self-contained**: Yes — all NPB kernels generate their own input based on "class" (problem size parameter).

### Priority: **HIGH** — Industry-standard HPC benchmarks, strong academic recognition.

**Recommended picks:** `npb_cg`, `npb_ep`, `npb_is` (3 benchmarks)

---

## Suite 5: ECP Proxy Applications

**Source:** https://proxyapps.exascaleproject.org/  
**Individual repos:**
- CoMD: https://github.com/ECP-copa/CoMD
- miniMD: https://github.com/Mantevo/miniMD
- PENNANT: https://github.com/lanl/PENNANT
- miniFE: https://github.com/Mantevo/miniFE

**Origin:** DOE Exascale Computing Project  
**Description:** Proxy applications that represent key computational patterns from full-scale HPC applications. Each proxy app captures the essential computation and communication patterns of a real application in a simplified, self-contained package.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `ecp_comd` | Classical molecular dynamics (Lennard-Jones) | Medium | Neighbor-list construction + force calculation. Self-contained with random atom positions. |
| `ecp_pennant` | Lagrangian hydrodynamics on unstructured meshes | High | ALE hydro code, mesh-based computation. |
| `ecp_minife` | Finite element assembly + CG solve | Medium | Sparse matrix assembly, CG solver. Classic HPC pattern. |
| `ecp_xsbench` | Monte Carlo neutron transport lookup | Medium | Binary search in energy grids, irregular memory access. |

### Portability Assessment
- **HIP**: Medium — some proxy apps already have CUDA/Kokkos GPU ports.
- **Metal**: Medium–High — kernels can be extracted but mesh/neighbor-list operations need careful porting.
- **Self-contained**: Yes — all proxy apps can generate synthetic problem setups.

### Priority: **MEDIUM** — High scientific value but higher implementation complexity.

**Recommended picks:** `ecp_comd`, `ecp_xsbench` (2 benchmarks)

---

## Suite 6: BabelStream

**Source:** https://github.com/UoB-HPC/BabelStream  
**Origin:** University of Bristol HPC group  
**Description:** A modern reimplementation of the STREAM benchmark for GPUs. Measures achievable memory bandwidth using Copy, Mul, Add, Triad, and Dot operations. Simple but essential for characterizing peak memory bandwidth — complements SHOC's triad with a more comprehensive memory bandwidth measurement.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `babelstream_triad` | Vector triad (a = b + scalar * c) | Low | Memory bandwidth baseline. |
| `babelstream_dot` | Dot product with reduction | Low | Tests reduction performance + bandwidth. |
| `babelstream_nstream` | Fused multiply-add stream | Low | N-stream variant with higher arithmetic intensity. |

### Portability Assessment
- **HIP**: Already supported natively in BabelStream.
- **Metal**: Low — simple memory-bound kernels translate directly.
- **Self-contained**: Yes — purely synthetic (array fill + compute).

### Priority: **HIGH** — Very easy to implement, essential memory bandwidth characterization.

**Recommended picks:** `babelstream_dot`, `babelstream_nstream` (2 benchmarks)

---

## Suite 7: Gunrock / GraphBIG-GPU

**Source:**
- Gunrock: https://github.com/gunrock/gunrock
- GraphBIG: https://github.com/graphbig/graphBIG

**Origin:** UC Davis (Gunrock), Georgia Tech (GraphBIG)  
**Description:** High-performance graph analytics frameworks for GPUs. Gunrock is the state-of-the-art GPU graph analytics library with optimized implementations of BFS, SSSP, PageRank, community detection, etc. GraphBIG provides a broader set of graph operations including graph construction and traversal.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `graph_pr` | PageRank (power iteration) | Medium | Iterative SpMV on graph adjacency. Different from BFS pattern. |
| `graph_cc` | Connected Components (label propagation) | Medium | Iterative label propagation until convergence. |
| `graph_tc` | Triangle Counting | Medium | Set intersection on neighbor lists. High compute + irregular memory. |
| `graph_bc` | Betweenness Centrality | High | Forward BFS + backward accumulation. |

### Portability Assessment
- **HIP**: Medium — extract individual kernels from Gunrock framework.
- **Metal**: Medium — CSR-based operations translate to compute shaders.
- **Self-contained**: Yes — R-MAT or Erdős–Rényi synthetic graph generators are simple to include.

### Priority: **MEDIUM** — Important workload class (graph analytics), but Gunrock API extraction requires care.

**Recommended picks:** `graph_pr`, `graph_tc` (2 benchmarks)

---

## Suite 8: CUDA SDK Extended Samples

**Source:** https://github.com/NVIDIA/cuda-samples  
**Origin:** NVIDIA  
**Description:** NVIDIA's official CUDA samples cover a wide range of GPU programming patterns. Beyond the already-implemented `transpose`, there are many self-contained benchmarks covering scan, histogram, convolution, sorting networks, and more.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `cuda_nbody` | N-body gravitational simulation | Low | Classic all-pairs force calculation. Tile-based shared memory optimization. |
| `cuda_convolution_separable` | Separable 2D convolution | Low | Row + column pass convolution. Tests shared memory + register usage. |
| `cuda_scan_large` | Large-array parallel prefix sum | Low | Work-efficient Blelloch scan. Fundamental GPU primitive. |
| `cuda_mersenne_twister` | Mersenne Twister PRNG | Low | Parallel random number generation. Tests register-heavy computation. |

### Portability Assessment
- **HIP**: Direct — CUDA samples hipify cleanly.
- **Metal**: Low — all are standard compute patterns.
- **Self-contained**: Yes — all generate synthetic data.

### Priority: **HIGH** — Very well-documented, easy to implement, covers fundamental patterns.

**Recommended picks:** `cuda_nbody`, `cuda_convolution_separable`, `cuda_scan_large` (3 benchmarks)

---

## Suite 9: Chai Benchmarks

**Source:** https://github.com/chai-benchmarks/chai  
**Origin:** University of Illinois  
**Description:** Chai (CPU-Heterogeneous-Accelerator Integration) benchmarks focus on collaborative CPU-GPU execution patterns. They measure how well workloads can be split and shared between CPU and GPU, covering task parallelism, producer-consumer patterns, and dynamic load balancing.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `chai_hsto` | Histogram (collaborative) | Low | Parallel histogram with atomic operations. Different strategy from Parboil's. |
| `chai_pad` | Padding removal/insertion | Low | Sparse-to-dense and dense-to-sparse transforms. Irregular scatter/gather. |
| `chai_cedd` | Canny Edge Detection | Medium | Multi-stage image processing pipeline: Gaussian blur → gradient → non-max suppression → hysteresis. |
| `chai_rscd` | Random Sample Consensus Detector | Medium | RANSAC-based model fitting, tests branch divergence and random access. |

### Portability Assessment
- **HIP**: Direct — kernels are straightforward CUDA.
- **Metal**: Low–Medium — pipeline stages are individual compute kernels.
- **Self-contained**: Yes — synthetic image/data generation.

### Priority: **MEDIUM** — Good workload diversity, moderate implementation effort.

**Recommended picks:** `chai_hsto`, `chai_cedd` (2 benchmarks)

---

## Suite 10: Tango Benchmark Suite

**Source:** https://github.com/yuhc/gpu-tango (academic)  
**Origin:** Academic research on GPU performance characterization  
**Description:** Tango provides targeted microbenchmarks and application kernels for GPU performance studies. It includes benchmarks that stress specific microarchitectural features like memory coalescing, bank conflicts, warp divergence, and occupancy effects.

### Recommended Benchmarks

| Benchmark | Description | Complexity | Notes |
|-----------|-------------|------------|-------|
| `tango_binomial_options` | Binomial option pricing (finance) | Medium | Tree traversal computation, tests register pressure. |
| `tango_blackscholes` | Black-Scholes option pricing | Low | Embarrassingly parallel math-heavy kernel. Classic GPU benchmark. |
| `tango_dxtc` | DXT texture compression | Medium | Image processing with parallel block compression. |

### Portability Assessment
- **HIP**: Direct — standard CUDA kernels.
- **Metal**: Low — math-heavy kernels translate directly.
- **Self-contained**: Yes — financial models use synthetic parameters, images generated procedurally.

### Priority: **MEDIUM** — Fills finance/image-processing niche.

**Recommended picks:** `tango_blackscholes`, `tango_binomial_options` (2 benchmarks)

---

## Summary: Recommended Implementation Plan

### Priority Ranking

| Priority | Suite | Recommended Count | Complexity | Rationale |
|----------|-------|-------------------|------------|-----------|
| 1 | **BabelStream** | 2 | Low | Trivial to implement, essential bandwidth characterization |
| 2 | **HeteroMark** | 3 | Low | Easy kernels, good heterogeneous coverage |
| 3 | **CUDA SDK Extended** | 3 | Low | Well-documented, fundamental patterns (nbody, conv, scan) |
| 4 | **Altis** | 3 | Medium | Modern suite, fills CFD/ray-tracing/particle gaps |
| 5 | **NPB-GPU** | 3 | Medium | Industry-standard HPC, strong academic value |
| 6 | **Lonestar GPU** | 2 | Medium | Irregular graph algorithms, unique workload patterns |
| 7 | **Graph Analytics** | 2 | Medium | PageRank + triangle counting fill graph analytics gap |
| 8 | **Chai** | 2 | Medium | Edge detection + histogram add image processing |
| 9 | **ECP Proxy Apps** | 2 | Medium–High | Scientific computing proxy apps |
| 10 | **Tango** | 2 | Low–Medium | Finance workloads fill niche |

### Total New Benchmarks: **24** (from 10 suites)

### Workload Coverage Gap Analysis

| Workload Pattern | Current Coverage | New Coverage |
|-----------------|------------------|--------------|
| Dense linear algebra | PolBench GEMM, SHOC GEMM, Parboil SGEMM | — (sufficient) |
| Sparse linear algebra | SHOC SpMV, Parboil SpMV | NPB CG |
| Stencils | SHOC stencil2d, Parboil stencil, PolyBench Jacobi/FDTD | Altis CFD |
| Graph traversal | Rodinia BFS | Lonestar SSSP, Graph CC |
| Graph analytics | — | HeteroMark PageRank, Graph PR, Graph TC |
| Sorting | SHOC sort | NPB IS |
| Reduction | SHOC reduction | BabelStream dot |
| Scan | SHOC scan | CUDA scan_large |
| FFT | SHOC FFT | NPB FT |
| N-body | — | CUDA nbody, Lonestar BH |
| Image processing | — | Chai CEDD, CUDA conv_separable |
| Monte Carlo/statistics | — | Altis particlefilter, NPB EP |
| Ray tracing | — | Altis raytracing |
| Encryption/bitwise | — | HeteroMark AES |
| Finance | — | Tango BlackScholes, binomial options |
| Signal processing | — | HeteroMark FIR |
| Memory bandwidth | SHOC triad, SHOC devicememory | BabelStream nstream |
| Molecular dynamics | — | ECP CoMD |
| Nuclear physics | — | ECP XSBench |

### Minimum Viable Set (8 suites, most impactful)

If only 8 suites are required, drop **Tango** and **Chai** (their workloads partially overlap with others). The remaining 8 suites provide 20 benchmarks with excellent diversity.

---

## Implementation Notes

### Build Pattern
All new benchmarks should follow the existing pattern:
- `{suite}_{benchmark}.hip` — HIP source (also compiled as CUDA via `hipcc` on NVIDIA)
- `{suite}_{benchmark}.metal` — Metal compute shader
- `{suite}_{benchmark}_metal.mm` — ObjC++ Metal host code
- `Makefile` — Auto-detecting CUDA/ROCm/Metal platform
- `README.md` — Algorithm description, build/run instructions, output format

### Output Format
All benchmarks must emit CSV on stdout:
```
kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
```

### Self-Contained Data
No benchmark should require external input files. All data must be:
- Synthetically generated at runtime (random arrays, procedural meshes)
- Generated from algorithmic parameters (NPB class sizes, graph generators)
- Embedded as constants (small lookup tables)

### Metal Portability Considerations
- No dynamic parallelism (use indirect dispatch or iterative host-side launches)
- No `printf` from device (use buffer-based reporting)
- No 64-bit atomics on older Metal GPUs (design around 32-bit atomics)
- Metal Shading Language uses `[[thread_position_in_grid]]` etc. instead of `threadIdx.x`
- Metal threadgroup memory ≈ CUDA shared memory

---

## Appendix: Source URLs

| Suite | Primary URL |
|-------|-------------|
| Lonestar GPU | https://github.com/IntelligentSoftwareSystems/Galois |
| Altis | https://github.com/utcs-scea/altis |
| HeteroMark | https://github.com/NUCAR-DEV/Hetero-Mark |
| NPB-GPU | https://github.com/GMAP/NPB-GPU |
| ECP CoMD | https://github.com/ECP-copa/CoMD |
| ECP XSBench | https://github.com/ANL-CESAR/XSBench |
| ECP miniFE | https://github.com/Mantevo/miniFE |
| ECP PENNANT | https://github.com/lanl/PENNANT |
| BabelStream | https://github.com/UoB-HPC/BabelStream |
| Gunrock | https://github.com/gunrock/gunrock |
| GraphBIG | https://github.com/graphbig/graphBIG |
| CUDA Samples | https://github.com/NVIDIA/cuda-samples |
| Chai | https://github.com/chai-benchmarks/chai |
| Tango | https://github.com/yuhc/gpu-tango |
