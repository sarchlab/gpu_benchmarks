#!/usr/bin/env python3
"""Generate a consolidated benchmark parameter-selection document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "parameter_selection.md"


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def format_values(values: list[Any]) -> str:
    return ", ".join(format_value(value) for value in values)


def format_non_scaling_defaults(params: list[dict[str, Any]]) -> str:
    if not params:
        return "`{}`"
    return "<br>".join(
        f"`{param['name']}` = `{format_value(param['default'])}`" for param in params
    )


def format_non_scaling_supported(params: list[dict[str, Any]]) -> str:
    if not params:
        return "`{}`"
    return "<br>".join(
        f"`{param['name']}`: `{format_values(param['values'])}`" for param in params
    )


def benchmark_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
    return int(item["tier"]), str(item["suite"]), str(item["benchmark"])


def load_benchmarks() -> list[dict[str, Any]]:
    benchmarks = []
    for path in sorted(REPO_ROOT.glob("tier*/*/params.json")):
        data = json.loads(path.read_text())
        data["_path"] = path.relative_to(REPO_ROOT).as_posix()
        benchmarks.append(data)
    return sorted(benchmarks, key=benchmark_sort_key)


def render_doc(benchmarks: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark Parameter Selection",
        "",
        "This document is generated from each benchmark's `params.json` file.",
        "It describes the default parameter selection used by `gpubench run`.",
        "",
        "Default behavior:",
        "",
        "- The scaling parameter is swept over every value listed in `scaling_param.values`.",
        "- Non-scaling parameters are swept over the Cartesian product of all listed `values`.",
        "- Passing `--sweep-non-scaling=false` uses each non-scaling parameter's `default` value instead.",
        "- Passing `--scaling-values` overrides the listed scaling values for that run only.",
        "",
        "Source docs: [params_format.md](params_format.md) and [cli_design.md](cli_design.md).",
        "",
    ]

    tiers = sorted({int(bench["tier"]) for bench in benchmarks})
    for tier in tiers:
        lines.extend(
            [
                f"## Tier {tier}",
                "",
                "| Benchmark | Suite | Scaling selection | Non-scaling selected defaults | Non-scaling supported values | Spec |",
                "|---|---|---|---|---|---|",
            ]
        )
        for bench in benchmarks:
            if int(bench["tier"]) != tier:
                continue
            scaling = bench["scaling_param"]
            non_scaling = bench["non_scaling_params"]
            scaling_cell = (
                f"`{scaling['name']}`: `{format_values(scaling['values'])}`"
            )
            defaults_cell = format_non_scaling_defaults(non_scaling)
            supported_cell = format_non_scaling_supported(non_scaling)
            spec_cell = f"[params.json](../{bench['_path']})"
            lines.append(
                "| "
                f"`{bench['benchmark']}` | "
                f"`{bench['suite']}` | "
                f"{scaling_cell} | "
                f"{defaults_cell} | "
                f"{supported_cell} | "
                f"{spec_cell} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    OUT_PATH.write_text(render_doc(load_benchmarks()))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
