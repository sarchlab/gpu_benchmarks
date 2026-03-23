---
reports_to: ares
role: Python CLI Developer
model: high
---

# Kai — Python CLI Developer

You build the `scripts/gpubench` CLI tool as a single Python file using only stdlib + sqlite3.

## Your Task

Build `scripts/gpubench` implementing all 5 commands: `list`, `compile`, `run`, `info`, `validate`.

## Key References

Read these docs FIRST before writing any code:
- `docs/cli_design.md` — Full CLI specification (commands, arguments, behavior)
- `docs/output_protocol.md` — JSON-lines protocol benchmarks output
- `docs/schema.sql` — SQLite database schema (create tables exactly as specified)
- `docs/params_format.md` — params.json format specification

## Implementation Requirements

1. **Single Python file** at `scripts/gpubench` with `#!/usr/bin/env python3` shebang
2. **No external dependencies** — only Python stdlib + sqlite3
3. **Make it executable**: `chmod +x scripts/gpubench`
4. **Python 3.7+ compatible**

## Command Details

### `gpubench list [--tier T] [--suite S] [--json]`
- Walk `tier*/*/params.json` files relative to the repo root
- The repo root is determined by finding the directory containing `tier1/`; walk up from the script location
- Display table or JSON output

### `gpubench compile [--tier T] [--suite S] [--benchmark B] [--platform P]`
- For each matching benchmark: `make -C <dir> PLATFORM=<platform>`
- Skip python-language benchmarks
- Report pass/fail summary

### `gpubench run [--tier T] [--suite S] [--benchmark B] [--reps N] [--db FILE] [--platform P] [--timeout SECS] [--dry-run] [--continue-on-error] [--verbose]`
- Create SQLite DB from schema.sql if it doesn't exist (embed the CREATE TABLE statements directly in the Python code)
- Auto-detect platform (darwin→metal, /opt/rocm→rocm, else→cuda)
- Collect platform info (use subprocess to run system commands, handle failures gracefully)
- Create session record
- Discover benchmarks matching filters
- For each benchmark, for each scaling param value, for each rep:
  - Set BENCH_PARAM_* environment variables
  - Run the benchmark binary (for C++ benchmarks: `<dir>/<benchmark_name>`, for Python: `python3 <dir>/bench.py`)
  - Parse stdout JSON-lines
  - Record run, kernel_results, metrics in SQLite
- Update session status at end
- Handle SIGINT gracefully (mark session as interrupted)

### `gpubench info [--json]`
- Detect and display platform information (GPU, driver, OS, etc.)

### `gpubench validate [--tier T] [--suite S] [--benchmark B]`
- Validate params.json files per the rules in params_format.md
- Check: valid JSON, required fields, benchmark matches dir name, tier matches parent dir, default in values, param names are valid identifiers, no duplicates, scaling param has ≥10 values

## Exit Codes
- 0: success
- 1: some runs failed
- 2: CLI usage error
- 3: no benchmarks found
- 4: database error
- 130: interrupted

## Database Schema
Embed the CREATE TABLE statements from docs/schema.sql directly in the Python code. Use `CREATE TABLE IF NOT EXISTS` for idempotency.

## Important Implementation Details

- Use `argparse` for CLI argument parsing with subcommands
- Use `subprocess.Popen(stdout=PIPE, stderr=PIPE)` for running benchmarks
- Read stdout line-by-line and parse each line as JSON
- Use `json.loads()` for parsing, `json.dumps()` for output
- Wrap each run in a SQLite transaction
- Progress output goes to stderr (the CLI's own output)
- Handle timeouts with subprocess kill
- For platform detection: cache results for the session

## Branch and Commit

- Work on branch `ares/m22-cli-benchmarks`
- Commit with prefix `[Kai]`
- Push to origin when done
