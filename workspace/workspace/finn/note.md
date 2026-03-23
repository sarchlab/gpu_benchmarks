# Finn — Workspace Notes

## Last completed: Issue #129 — NPB-GPU benchmarks (ep + is)

- Created `tier2/npb_ep/` and `tier2/npb_is/` on branch `ares/m19-tier2-suites1`
- Each has 5 files: .hip, _metal.mm, .metal, Makefile, README.md
- Both compile and pass verification on Metal (Apple M2)
- Used integer-based LCG RNG for cross-platform consistency (float-based LCG had precision issues on Metal)
- Commits: `[Finn] Add tier2/npb_ep`, `[Finn] Add tier2/npb_is`
- Pushed to origin
