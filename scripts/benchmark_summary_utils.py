from __future__ import annotations

import glob
import json
import math
import os


def load_result_files(results_dir: str, patterns: tuple[str, ...]) -> tuple[list[dict], list[str]]:
    files: list[str] = []
    for pattern in patterns:
        files = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if files:
            break
    if not files:
        joined_patterns = ", ".join(patterns)
        raise FileNotFoundError(
            f"No benchmark result files found under {results_dir} matching: {joined_patterns}"
        )

    payloads = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads, files


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((value - mean_val) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_summary(summary_output: str, summary: dict) -> None:
    summary_dir = os.path.dirname(summary_output)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
