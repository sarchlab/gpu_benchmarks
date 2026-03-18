#!/usr/bin/env python3
"""
report.py — Generate a markdown summary from GPU benchmark results.

Usage:
    python3 scripts/report.py <results_dir>

    <results_dir>: directory containing <tier>_<benchmark>.csv files
                   (produced by scripts/run_all.sh)

Output:
    Prints report to stdout AND writes <results_dir>/report.md
"""

import sys
import os
import csv
import glob
from pathlib import Path
from datetime import datetime


def parse_csv_file(filepath):
    """
    Parse a benchmark CSV file. Returns list of row dicts with normalized fields:
      - benchmark: str
      - size: str
      - time_ms: float or None
      - metric: str (formatted string, may include units from context)
    """
    rows = []
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        if not content:
            return rows

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')

            if len(parts) == 4:
                # 4-field format: benchmark_name,size,time_ms,metric_value
                try:
                    rows.append({
                        'benchmark': parts[0].strip(),
                        'size': parts[1].strip(),
                        'time_ms': float(parts[2]),
                        'metric': parts[3].strip(),
                    })
                except (ValueError, IndexError):
                    pass
            elif len(parts) == 7:
                # 7-field format: kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
                try:
                    rows.append({
                        'benchmark': parts[0].strip(),
                        'size': parts[1].strip(),
                        'time_ms': float(parts[3]),  # avg_ms
                        'metric': f"min={parts[4]},max={parts[5]},std={parts[6].strip()}",
                    })
                except (ValueError, IndexError):
                    pass
            else:
                # Skip malformed lines silently
                pass
    except Exception:
        pass  # Return empty rows for unreadable files
    return rows


def generate_report(results_dir):
    results_dir = Path(results_dir)
    csv_files = sorted(results_dir.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    lines = []
    lines.append(f"# GPU Benchmark Results")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Results directory:** `{results_dir}`")
    lines.append(f"")

    # Group by tier
    tier_data = {}
    for csv_path in csv_files:
        stem = csv_path.stem  # e.g. "tier1_memory_bandwidth"
        parts = stem.split('_', 1)
        if len(parts) == 2:
            tier, bench = parts
        else:
            tier, bench = 'unknown', stem

        rows = parse_csv_file(csv_path)
        if tier not in tier_data:
            tier_data[tier] = []
        tier_data[tier].append((bench, rows))

    total_benchmarks = 0
    total_rows = 0

    for tier in sorted(tier_data.keys()):
        lines.append(f"## {tier.upper()}")
        lines.append(f"")
        lines.append(f"| Benchmark | Size | Time (ms) | Metric |")
        lines.append(f"|-----------|------|-----------|--------|")

        for bench, rows in sorted(tier_data[tier]):
            if not rows:
                lines.append(f"| {bench} | — | — | (no output) |")
                total_benchmarks += 1
            else:
                for row in rows:
                    lines.append(
                        f"| {row['benchmark']} | {row['size']} | "
                        f"{row['time_ms']:.3f} | {row['metric']} |"
                    )
                total_benchmarks += 1
                total_rows += len(rows)

        lines.append(f"")

    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"- **Benchmarks run:** {total_benchmarks}")
    lines.append(f"- **Result rows:** {total_rows}")
    lines.append(f"")

    report_text = '\n'.join(lines)

    # Print to stdout
    print(report_text)

    # Write to file
    report_path = results_dir / 'report.md'
    report_path.write_text(report_text)
    print(f"\n✓ Report written to: {report_path}", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)
    generate_report(sys.argv[1])
