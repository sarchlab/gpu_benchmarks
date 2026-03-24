#!/usr/bin/env python3
"""Static policy checker for Tier1/Tier2 scaling parameters.

Policy:
1) Every Tier1/Tier2 benchmark must include at least 3 non-power-of-two
   scaling values.
2) Tier1 per-thread-op throughput benchmarks must have max scaling value <= 65536.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]

TIER1_THROUGHPUT_CAP = {
    "atomic_operations": 65536,
    "warp_shuffle": 65536,
    "fp16_throughput": 65536,
    "fp32_throughput": 65536,
    "fp64_throughput": 65536,
    "int32_throughput": 65536,
}


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def non_power_of_two_count(values: Iterable[int]) -> int:
    return sum(1 for v in values if isinstance(v, int) and v > 0 and not is_power_of_two(v))


def load_values(path: Path) -> Tuple[List[int], List[str]]:
    errors: List[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        return [], [f"{path}: invalid JSON ({exc})"]

    scaling_param = data.get("scaling_param")
    if not isinstance(scaling_param, dict):
        return [], [f"{path}: missing scaling_param object"]

    values = scaling_param.get("values")
    if not isinstance(values, list) or not values:
        return [], [f"{path}: scaling_param.values must be a non-empty list"]

    if any(not isinstance(v, int) or v <= 0 for v in values):
        errors.append(f"{path}: scaling_param.values must contain only positive integers")

    return values, errors


def check() -> int:
    failures: List[str] = []
    checked = 0

    tier1_files = sorted((ROOT / "tier1").glob("*/params.json"))
    tier2_files = sorted((ROOT / "tier2").glob("*/params.json"))

    for path in [*tier1_files, *tier2_files]:
        checked += 1
        values, errors = load_values(path)
        failures.extend(errors)
        if errors:
            continue

        non_pot = non_power_of_two_count(values)
        if non_pot < 3:
            failures.append(
                f"{path}: scaling_param.values has {non_pot} non-power-of-two values (<3)"
            )

    for benchmark, cap in TIER1_THROUGHPUT_CAP.items():
        path = ROOT / "tier1" / benchmark / "params.json"
        values, errors = load_values(path)
        failures.extend(errors)
        if errors:
            continue

        max_value = max(values)
        if max_value > cap:
            failures.append(
                f"{path}: max scaling value {max_value} exceeds Tier1 cap {cap}"
            )

    print(
        "Checked policy on "
        f"{checked} benchmarks (tier1={len(tier1_files)}, tier2={len(tier2_files)})."
    )

    if failures:
        print(f"FAILED: {len(failures)} policy violation(s) detected:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASS: Tier1/Tier2 scaling policy checks succeeded.")
    print("  - Each benchmark has >=3 non-power-of-two scaling values.")
    print("  - Tier1 throughput-style per-thread-op scaling max is <= 65536.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
