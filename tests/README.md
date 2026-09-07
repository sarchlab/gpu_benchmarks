# GPU regression checks

`finite_verification.py` exercises the actual verification paths of 27 Tier 2
benchmarks. It makes isolated source copies, injects values immediately before
the existing comparisons, and builds those copies with the benchmark Makefiles.
The production sources and executables receive no injection interface.

```sh
python3 tests/finite_verification.py --platform cuda --work-dir /tmp/finite-cuda
python3 tests/finite_verification.py --platform rocm --work-dir /tmp/finite-rocm
```

Use a new output directory for each run. Select a subset by repeating
`--benchmark NAME`. `--source PATH` selects a checkout for before/after checks.
`--compile-only` checks the injection anchors and builds the test executables
without claiming GPU verification; CI uses this mode because its runner has no
GPU. Runtime checks require the selected compiler and a compatible GPU.

Each compared output/reference pair receives a single NaN, positive infinity,
and negative infinity, independently. An incorrect finite output control affects
all compared samples so it exceeds Barnes–Hut's existing approximation allowance.
A normal finite run must still pass. FFT checks both float and double precision;
Parboil SpMV checks its host self-test and actual GPU result. Complex/vector
comparisons cover each compared component independently. CG/MG injection targets
the reduced residual and reference norms that their verifiers actually compare.

Failure requires the hook to have executed, a verification failure diagnostic,
and exit status 1, with no HIP runtime error. A crash, timeout, missing hook,
or later cleanup error cannot satisfy the test. A timeout aborts the run; the
last completed benchmark report may therefore be partial.
Logs and `report.json` retain each result and the original source hash. The
minimum declared scaling value and existing non-scaling defaults are used, with
verification enabled and the clock/cache warmer disabled to keep these functional
checks short. These are verification tests, not performance measurements.

The R-04 audit fixes finite comparisons and error propagation, preserving existing
tolerances, sampled regions, and measured work. It also checks for non-finite
intermediate error metrics where normalization or max accumulation could hide
invalid values. One non-finite Barnes–Hut sample fails regardless of the quota
for finite approximation errors. SHOC reduction uses the same pass condition for
its diagnostic and exit status.

The remaining existing floating-point predicates in Rodinia LUD/Gaussian,
NPB EP's reference comparison, and Altis
particle filter use positive acceptance comparisons; those comparisons reject
NaN. Parboil LBM checks NaN and density bounds. Integer/byte verifiers have no
non-finite values. Missing verification, unchecked output components, unverified
NPB EP sizes, and verification on a separate small workload remain coverage
issues under R-13; this test does not establish their correctness.

CUDA execution and ROCm execution must be reported separately. A CUDA pass or
ROCm compile alone does not establish AMD runtime qualification (R-19).
