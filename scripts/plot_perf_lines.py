#!/usr/bin/env python3
"""Plot performance lines with confidence intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "seaborn is required to draw confidence ranges. "
        "Install plotting dependencies with: pip install -r requirements.txt"
    ) from exc


TRACE_COLORS = {
    "trace collected": "#1b9e77",
    "trace not collected": "#d95f02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one chart per benchmark/non-scaling parameter combination. "
            "X is the scaling parameter, Y is total kernel time, and the "
            "shaded band is seaborn's 95% confidence interval across repeated "
            "runs."
        )
    )
    parser.add_argument("--db", default="results.db", help="gpubench SQLite DB")
    parser.add_argument(
        "--trace-root",
        default="/scratch/data/yifan/new_traces",
        help="root directory containing collected traces",
    )
    parser.add_argument(
        "--out-dir", default="perf_charts", help="directory for PNG charts"
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "chart"


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def config_dir(
    scaling_name: str, scaling_value: str, non_scaling_params_json: str
) -> str:
    parts = [f"{scaling_name}_{scaling_value}"]
    try:
        params = json.loads(non_scaling_params_json or "{}")
    except json.JSONDecodeError:
        params = {}
    for key in sorted(params):
        parts.append(f"{key}_{format_value(params[key])}")
    return "__".join(parts)


def short_params(params_json: str) -> str:
    try:
        params = json.loads(params_json or "{}")
    except json.JSONDecodeError:
        return params_json
    if not params:
        return "default"
    return ", ".join(f"{k}={format_value(v)}" for k, v in sorted(params.items()))


def trace_name_from_ref(ref: str) -> str | None:
    name = Path(ref.strip()).name
    if name.endswith(".trace.zsim"):
        return name[: -len(".trace.zsim")] + ".trace.zst"
    if name.endswith(".trace.zst"):
        return name
    return None


def trace_complete(trace_cfg_dir: Path) -> bool:
    traces_dir = trace_cfg_dir / "traces"
    if (traces_dir / "trace_timeout.json").exists() or not traces_dir.is_dir():
        return False

    names = {p.name for p in traces_dir.iterdir() if p.is_file()}
    kernels_lists = sorted(
        n for n in names if n == "kernelslist" or n.startswith("kernelslist_")
    )
    stats_files = {n for n in names if n == "stats" or n.startswith("stats_")}
    trace_files = {n for n in names if n.endswith(".trace.zst")}
    if not kernels_lists or not stats_files or not trace_files:
        return False

    for kernels_list in kernels_lists:
        suffix = kernels_list.removeprefix("kernelslist")
        if suffix and f"stats{suffix}" not in stats_files:
            continue

        refs: list[str] = []
        try:
            lines = (traces_dir / kernels_list).read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            for field in line.replace(",", " ").split():
                trace_name = trace_name_from_ref(field)
                if trace_name:
                    refs.append(trace_name)

        if refs and all(ref in trace_files for ref in refs):
            return True
    return False


def load_runs(db_path: Path) -> pd.DataFrame:
    # Benchmarks with in-program measurement loops store one kernel_results
    # row per timed iteration. Average within each (run, kernel) before
    # summing across kernels, so a run's total means "one pass over every
    # kernel" regardless of how many iterations were recorded — keeping new
    # per-iteration runs comparable to older single-row-per-kernel runs in
    # the same database.
    query = """
        WITH kernel_totals AS (
            SELECT run_id, SUM(avg_ms) AS total_kernel_time_ms
            FROM (
                SELECT run_id, kernel_name, AVG(time_ms) AS avg_ms
                FROM kernel_results
                GROUP BY run_id, kernel_name
            )
            GROUP BY run_id
        )
        SELECT
            r.tier,
            r.suite,
            r.benchmark,
            r.scaling_param_name,
            r.scaling_param_value,
            r.non_scaling_params_json,
            COALESCE(k.total_kernel_time_ms, r.total_time_ms) AS total_kernel_time_ms,
            CASE
                WHEN k.total_kernel_time_ms IS NOT NULL THEN 'kernel_results'
                ELSE 'runs.total_time_ms'
            END AS timing_source
        FROM runs r
        LEFT JOIN kernel_totals k ON k.run_id = r.id
        WHERE r.status = 'ok'
          AND COALESCE(k.total_kernel_time_ms, r.total_time_ms) IS NOT NULL
        ORDER BY r.tier, r.suite, r.benchmark, r.scaling_param_value
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def annotate_trace_status(runs: pd.DataFrame, trace_root: Path) -> pd.DataFrame:
    status_cache: dict[tuple[str, str], bool] = {}
    statuses = []
    trace_dirs = []

    for row in runs.itertuples(index=False):
        cfg = config_dir(
            row.scaling_param_name,
            str(row.scaling_param_value),
            row.non_scaling_params_json,
        )
        trace_dir = trace_root / row.benchmark / cfg
        key = (row.benchmark, cfg)
        complete = status_cache.get(key)
        if complete is None:
            complete = trace_complete(trace_dir)
            status_cache[key] = complete
        statuses.append("trace collected" if complete else "trace not collected")
        trace_dirs.append(str(trace_dir))

    runs = runs.copy()
    runs["trace_status"] = statuses
    runs["trace_dir"] = trace_dirs
    return runs


def prepare_group(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    agg = (
        group.groupby("scaling_param_value", as_index=False)
        .agg(
            mean_ms=("total_kernel_time_ms", "mean"),
            std_ms=("total_kernel_time_ms", "std"),
            reps=("total_kernel_time_ms", "count"),
            trace_status=("trace_status", "first"),
            timing_source=("timing_source", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index(drop=True)
    )

    plot_data = group.copy()
    numeric = pd.to_numeric(agg["scaling_param_value"], errors="coerce")
    if numeric.notna().all():
        agg["x"] = numeric
        plot_data["x"] = pd.to_numeric(plot_data["scaling_param_value"])
        agg = agg.sort_values("x")
        plot_data = plot_data.sort_values("x")
        x_kind = "numeric"
    else:
        agg["x"] = agg["scaling_param_value"].astype(str)
        plot_data["x"] = plot_data["scaling_param_value"].astype(str)
        agg = agg.sort_values("x", key=lambda col: col.map(str))
        ordered = agg["x"].tolist()
        plot_data["x"] = pd.Categorical(plot_data["x"], categories=ordered, ordered=True)
        plot_data = plot_data.sort_values("x")
        x_kind = "categorical"
    return agg, plot_data, x_kind


def plot_group(group: pd.DataFrame, out_path: Path) -> dict[str, object]:
    first = group.iloc[0]
    agg, plot_data, x_kind = prepare_group(group)
    title = f"tier{first.tier}/{first.suite}/{first.benchmark}"
    params = short_params(first.non_scaling_params_json)

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    sns.lineplot(
        data=plot_data,
        x="x",
        y="total_kernel_time_ms",
        estimator="mean",
        errorbar=("ci", 95),
        color="#e84a35",
        marker="o",
        markersize=6,
        linewidth=2.2,
        ax=ax,
        zorder=2,
    )

    ax.set_title(title)
    ax.set_xlabel(str(first.scaling_param_name))
    ax.set_ylabel("total kernel time (ms)")
    ax.grid(True, alpha=0.35)

    subtitle = textwrap.fill(f"non-scaling: {params}", width=95)
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=8)

    if x_kind == "categorical":
        ax.tick_params(axis="x", rotation=35)
    elif len(agg) > 12:
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "chart": str(out_path),
        "tier": int(first.tier),
        "suite": first.suite,
        "benchmark": first.benchmark,
        "scaling_param": first.scaling_param_name,
        "non_scaling_params": first.non_scaling_params_json,
        "points": int(len(agg)),
        "trace_collected_points": int((agg["trace_status"] == "trace collected").sum()),
        "trace_not_collected_points": int(
            (agg["trace_status"] == "trace not collected").sum()
        ),
        "runs": int(agg["reps"].sum()),
        "timing_source": ",".join(sorted(set(group["timing_source"]))),
        "min_mean_ms": float(agg["mean_ms"].min()),
        "max_mean_ms": float(agg["mean_ms"].max()),
    }


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    trace_root = Path(args.trace_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(db_path)
    if runs.empty:
        raise SystemExit(f"No successful timing rows found in {db_path}")
    runs = annotate_trace_status(runs, trace_root)

    manifest = []
    group_cols = [
        "tier",
        "suite",
        "benchmark",
        "scaling_param_name",
        "non_scaling_params_json",
    ]
    for _, group in runs.groupby(group_cols, sort=True):
        first = group.iloc[0]
        digest = hashlib.sha1(
            str(first.non_scaling_params_json).encode("utf-8")
        ).hexdigest()[:8]
        filename = safe_name(f"tier{first.tier}_{first.suite}_{first.benchmark}_{digest}.png")
        manifest.append(plot_group(group, out_dir / filename))

    manifest_df = pd.DataFrame(manifest).sort_values(["tier", "suite", "benchmark"])
    manifest_path = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    points = int(manifest_df["points"].sum())
    traced = int(manifest_df["trace_collected_points"].sum())
    print(f"wrote {len(manifest_df)} charts to {out_dir}")
    print(f"wrote manifest to {manifest_path}")
    print(f"trace-collected points: {traced}/{points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
