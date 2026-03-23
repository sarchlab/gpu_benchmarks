# Jade — Workspace Notes

## 2026-03-19: Completed Tier 2 params.json (Issue #147)

### What I did
- Analyzed all 51 Tier 2 .hip source files across 13 suites
- Created params.json for each benchmark directory with:
  - Scaling parameter (16-22 values each, spanning small to large)
  - Non-scaling parameters (3-5 values each) with defaults
  - Kernel names extracted from actual `__global__` function declarations
- Updated .gitignore to allow `tier2/*/params.json`
- Committed to branch `jade/params-tier2` and pushed

### Suites covered (51 benchmarks):
- SHOC (9): triad, reduction, scan, gemm, devicememory, sort, spmv, stencil2d, fft
- PolyBench (9): gemm, 2dconv, fdtd2d, 3mm, atax, bicg, mvt, gramschmidt, jacobi2d
- Rodinia (9): hotspot, bfs, nw, srad, gaussian, lud, pathfinder, kmeans, backprop
- Parboil (4): sgemm, spmv, stencil, histogram
- CUDA SDK (4): transpose, nbody, convolution_separable, scan_large
- HeteroMark (3): aes, fir, pagerank
- Tango (2): blackscholes, binomial_options
- NPB-GPU (2): ep, is
- Altis (3): cfd, particlefilter, raytracing
- Lonestar (2): sssp, bh
- Graph (2): pr, tc
- Chai (2): hsto, cedd

### Previous work
- 2026-03-19: Completed Tier 2 Research (Issue #102) — documented 10 suites with 24 benchmarks
