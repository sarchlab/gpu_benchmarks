# scripts/

## gpubench

The `gpubench` CLI tool discovers, compiles, runs, and validates GPU benchmarks. Results are stored in an SQLite database. No pip dependencies — Python 3.7+ stdlib only.

```bash
# List all benchmarks
python3 scripts/gpubench list

# Compile C++ benchmarks
python3 scripts/gpubench compile

# Run benchmarks (results → results.db)
python3 scripts/gpubench run

# Validate params.json files
python3 scripts/gpubench validate

# Show platform info
python3 scripts/gpubench info
```

See [`docs/cli_design.md`](../docs/cli_design.md) for the full command reference.
