# Vera — Workspace Notes

## Last Cycle

Completed issue #148: CLI architecture design docs.

### Delivered (branch `vera/cli-design`, commit `3670c4f`):
- `docs/params_format.md` — Full params.json spec with validation rules, type system, examples for tier 1/2/5
- `docs/output_protocol.md` — JSON-lines stdout protocol (kernel, summary, error, info message types)
- `docs/schema.sql` — SQLite schema: platforms, sessions, runs, kernel_results, metrics tables with indexes and 7 example queries
- `docs/cli_design.md` — CLI commands (list, compile, run, info, validate), platform detection, parameter sweeping, execution flow

### Design Decisions:
- Parameters passed via `BENCH_PARAM_{name}` environment variables
- Single Python file CLI at `scripts/gpubench`
- Sessions table groups runs from a single CLI invocation
- Platform fingerprint deduplication via unique index
- Default mode sweeps scaling param only; `--sweep-non-scaling` for Cartesian product
- Exit code conventions: 0=success, 1=partial fail, 2=usage, 3=no benchmarks, 4=db error, 130=interrupted
