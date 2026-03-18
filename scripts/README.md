# scripts/

Utility scripts for running benchmarks and collecting results.

## run_all.sh

Runs all benchmarks in the suite and captures CSV output to `results/<timestamp>/`.

```bash
# Auto-detect platform
bash scripts/run_all.sh

# Force CUDA
bash scripts/run_all.sh cuda

# Force ROCm
bash scripts/run_all.sh rocm

# Force Metal (Apple)
bash scripts/run_all.sh metal

# Custom timeout (seconds per benchmark, default 300)
bash scripts/run_all.sh cuda 120
```

## report.py

Generates a markdown summary from collected results.

```bash
# Run after run_all.sh
python3 scripts/report.py results/<timestamp>/

# Or use make
make run-all    # runs all benchmarks
make report     # generates report from latest results
```
