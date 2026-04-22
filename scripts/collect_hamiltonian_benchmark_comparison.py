from __future__ import annotations

import argparse
import json
import os
from typing import Any

BIN_ORDER = ["high", "mid", "low"]
BIN_LABELS = {
    "high": "high",
    "mid": "medium",
    "low": "low",
}

METRIC_SPECS = {
    "agent_success": {
        "summary_mean_key": "agent_success_mean",
        "summary_std_key": "agent_success_std",
        "higher_is_better": True,
    },
    "hamiltonian_path_success": {
        "summary_mean_key": "hamiltonian_path_success_mean",
        "summary_std_key": "hamiltonian_path_success_std",
        "higher_is_better": True,
    },
    "postprocessed_plan_length_rel_error": {
        "summary_mean_key": "postprocessed_plan_length_rel_error_mean",
        "summary_std_key": "postprocessed_plan_length_rel_error_std",
        "higher_is_better": False,
    },
}

OVERALL_METRIC_KEYS = {
    "agent_success": ("final_agent_success_mean", "final_agent_success_std"),
    "hamiltonian_path_success": (
        "final_hamiltonian_success_mean",
        "final_hamiltonian_success_std",
    ),
    "postprocessed_plan_length_rel_error": (
        "final_postprocessed_rel_error_mean",
        "final_postprocessed_rel_error_std",
    ),
}

TASK_METRIC_KEYS = {
    "agent_success": (
        "agent_success_mean",
        "agent_success_std_over_repeats",
    ),
    "hamiltonian_path_success": (
        "hamiltonian_path_success_mean",
        "hamiltonian_path_success_std_over_repeats",
    ),
    "postprocessed_plan_length_rel_error": (
        "postprocessed_plan_length_rel_error_mean",
        "postprocessed_plan_length_rel_error_std_over_repeats",
    ),
}


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for task_result in summary.get("task_results", []):
        out[int(task_result["task_id"])] = task_result
    return out


def _difficulty_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = summary.get("bin_cross_task_results", {})
    return {str(name): value for name, value in raw.items()}


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _delta(online_value: float | None, baseline_value: float | None) -> float | None:
    if online_value is None or baseline_value is None:
        return None
    return float(online_value) - float(baseline_value)


def _better_variant(
    online_value: float | None,
    baseline_value: float | None,
    *,
    higher_is_better: bool,
) -> str | None:
    if online_value is None or baseline_value is None:
        return None
    if online_value == baseline_value:
        return "tie"
    if higher_is_better:
        return "online" if online_value > baseline_value else "baseline"
    return "online" if online_value < baseline_value else "baseline"


def _build_metric_comparison(
    *,
    online_mean: Any,
    baseline_mean: Any,
    higher_is_better: bool,
    online_std: Any = None,
    baseline_std: Any = None,
    online_n: Any = None,
    baseline_n: Any = None,
) -> dict[str, Any]:
    online_mean_f = _to_float_or_none(online_mean)
    baseline_mean_f = _to_float_or_none(baseline_mean)
    online_std_f = _to_float_or_none(online_std)
    baseline_std_f = _to_float_or_none(baseline_std)
    return {
        "online_mean": online_mean_f,
        "baseline_mean": baseline_mean_f,
        "online_std": online_std_f,
        "baseline_std": baseline_std_f,
        "online_n": None if online_n is None else int(online_n),
        "baseline_n": None if baseline_n is None else int(baseline_n),
        "delta_online_minus_baseline": _delta(online_mean_f, baseline_mean_f),
        "better_variant": _better_variant(
            online_mean_f,
            baseline_mean_f,
            higher_is_better=higher_is_better,
        ),
        "higher_is_better": bool(higher_is_better),
    }


def _build_overall_comparison(
    online: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric_name, (mean_key, std_key) in OVERALL_METRIC_KEYS.items():
        out[metric_name] = _build_metric_comparison(
            online_mean=online.get(mean_key),
            baseline_mean=baseline.get(mean_key),
            online_std=online.get(std_key),
            baseline_std=baseline.get(std_key),
            online_n=len(online.get("repeat_results", [])),
            baseline_n=len(baseline.get("repeat_results", [])),
            higher_is_better=METRIC_SPECS[metric_name]["higher_is_better"],
        )
    return out


def _build_task_comparisons(
    online: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    online_tasks = _task_map(online)
    baseline_tasks = _task_map(baseline)
    common_task_ids = sorted(set(online_tasks) & set(baseline_tasks))
    comparisons: list[dict[str, Any]] = []
    for task_id in common_task_ids:
        online_task = online_tasks[task_id]
        baseline_task = baseline_tasks[task_id]
        task_comp: dict[str, Any] = {"task_id": int(task_id)}
        for metric_name, (mean_key, std_key) in TASK_METRIC_KEYS.items():
            task_comp[metric_name] = _build_metric_comparison(
                online_mean=online_task.get(mean_key),
                baseline_mean=baseline_task.get(mean_key),
                online_std=online_task.get(std_key),
                baseline_std=baseline_task.get(std_key),
                online_n=len(online_task.get("repeat_hamiltonian_scores", [])),
                baseline_n=len(baseline_task.get("repeat_hamiltonian_scores", [])),
                higher_is_better=METRIC_SPECS[metric_name]["higher_is_better"],
            )
        comparisons.append(task_comp)
    return comparisons


def _build_difficulty_comparisons(
    online: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    online_bins = _difficulty_map(online)
    baseline_bins = _difficulty_map(baseline)
    common_bins = [name for name in BIN_ORDER if name in online_bins and name in baseline_bins]
    comparisons: list[dict[str, Any]] = []
    for difficulty in common_bins:
        online_bin = online_bins[difficulty]
        baseline_bin = baseline_bins[difficulty]
        diff_comp: dict[str, Any] = {
            "difficulty": difficulty,
            "difficulty_label": BIN_LABELS.get(difficulty, difficulty),
        }
        for metric_name, spec in METRIC_SPECS.items():
            diff_comp[metric_name] = _build_metric_comparison(
                online_mean=online_bin.get(spec["summary_mean_key"]),
                baseline_mean=baseline_bin.get(spec["summary_mean_key"]),
                online_std=online_bin.get(spec["summary_std_key"]),
                baseline_std=baseline_bin.get(spec["summary_std_key"]),
                online_n=online_bin.get("n_samples"),
                baseline_n=baseline_bin.get("n_samples"),
                higher_is_better=spec["higher_is_better"],
            )
        comparisons.append(diff_comp)
    return comparisons


def _build_task_difficulty_comparisons(
    online: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    online_tasks = _task_map(online)
    baseline_tasks = _task_map(baseline)
    common_task_ids = sorted(set(online_tasks) & set(baseline_tasks))
    comparisons: list[dict[str, Any]] = []
    for task_id in common_task_ids:
        online_bins = online_tasks[task_id].get("bin_results", {})
        baseline_bins = baseline_tasks[task_id].get("bin_results", {})
        common_bins = [name for name in BIN_ORDER if name in online_bins and name in baseline_bins]
        for difficulty in common_bins:
            online_bin = online_bins[difficulty]
            baseline_bin = baseline_bins[difficulty]
            item: dict[str, Any] = {
                "task_id": int(task_id),
                "difficulty": difficulty,
                "difficulty_label": BIN_LABELS.get(difficulty, difficulty),
            }
            for metric_name, spec in METRIC_SPECS.items():
                item[metric_name] = _build_metric_comparison(
                    online_mean=online_bin.get(spec["summary_mean_key"]),
                    baseline_mean=baseline_bin.get(spec["summary_mean_key"]),
                    online_std=online_bin.get(spec["summary_std_key"]),
                    baseline_std=baseline_bin.get(spec["summary_std_key"]),
                    online_n=online_bin.get("n_samples"),
                    baseline_n=baseline_bin.get("n_samples"),
                    higher_is_better=spec["higher_is_better"],
                )
            comparisons.append(item)
    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Compare online and fixed-temporal Hamiltonian benchmark summaries."
    )
    parser.add_argument("--online_summary", required=True)
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    online = _load_json(args.online_summary)
    baseline = _load_json(args.baseline_summary)

    overall_comparison = _build_overall_comparison(online, baseline)
    task_comparisons = _build_task_comparisons(online, baseline)
    difficulty_comparisons = _build_difficulty_comparisons(online, baseline)
    task_difficulty_comparisons = _build_task_difficulty_comparisons(online, baseline)

    summary = {
        "dataset": online.get("dataset") or baseline.get("dataset"),
        "run_timestamp": online.get("run_timestamp") or baseline.get("run_timestamp"),
        "online_summary": os.path.abspath(args.online_summary),
        "baseline_summary": os.path.abspath(args.baseline_summary),
        "online_planner_variants": online.get("planner_variants", []),
        "baseline_planner_variants": baseline.get("planner_variants", []),
        "delta_convention": "online_minus_baseline",
        "comparison_axes": ["overall", "task", "difficulty", "task_difficulty"],
        "metric_specs": {
            name: {
                "higher_is_better": spec["higher_is_better"],
            }
            for name, spec in METRIC_SPECS.items()
        },
        "overall_comparison": overall_comparison,
        "task_comparisons": task_comparisons,
        "difficulty_comparisons": difficulty_comparisons,
        "task_difficulty_comparisons": task_difficulty_comparisons,
        # Backward-compatible convenience fields.
        "online_final_hamiltonian_success_mean": online.get("final_hamiltonian_success_mean"),
        "baseline_final_hamiltonian_success_mean": baseline.get(
            "final_hamiltonian_success_mean"
        ),
        "online_final_postprocessed_rel_error_mean": online.get(
            "final_postprocessed_rel_error_mean"
        ),
        "baseline_final_postprocessed_rel_error_mean": baseline.get(
            "final_postprocessed_rel_error_mean"
        ),
        "final_route_success_delta_online_minus_baseline": overall_comparison[
            "hamiltonian_path_success"
        ]["delta_online_minus_baseline"],
        "final_rel_error_delta_online_minus_baseline": overall_comparison[
            "postprocessed_plan_length_rel_error"
        ]["delta_online_minus_baseline"],
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"comparison_summary: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
