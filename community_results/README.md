# Community Benchmark Results

This directory contains GPU benchmark results submitted by the community.

## How to Submit

1. Run benchmarks using `gpubench run`.
2. Export your results:
   ```bash
   gpubench export --db results.db > community_results/<gpu_model>_<platform>_<date>.json
   ```
3. Follow the naming convention:
   ```
   <gpu_model>_<platform>_<YYYY-MM-DD>.json
   ```
   Examples:
   - `rtx4090_cuda_2024-03-15.json`
   - `rx7900xtx_rocm_2024-03-15.json`
   - `m2ultra_metal_2024-03-15.json`
4. Open a Pull Request with your file.

## Guidelines

- Run benchmarks on an **idle system** for consistent results.
- Use the default repetitions (7).
- Do not modify the exported JSON format.
- One file per GPU/platform/date combination.

For full details, see [docs/result_sharing.md](../docs/result_sharing.md).
