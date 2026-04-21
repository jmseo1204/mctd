from __future__ import annotations

import argparse
import json
import os


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_map(summary: dict) -> dict[int, dict]:
    out = {}
    for task_result in summary.get("task_results", []):
        out[int(task_result["task_id"])] = task_result
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare online and fixed-temporal Hamiltonian benchmark summaries.")
    parser.add_argument("--online_summary", required=True)
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    online = _load_json(args.online_summary)
    baseline = _load_json(args.baseline_summary)

    online_tasks = _task_map(online)
    baseline_tasks = _task_map(baseline)
    common_task_ids = sorted(set(online_tasks) & set(baseline_tasks))

    per_task = []
    for task_id in common_task_ids:
        online_task = online_tasks[task_id]
        baseline_task = baseline_tasks[task_id]
        online_rel = online_task.get("postprocessed_plan_length_rel_error_mean")
        baseline_rel = baseline_task.get("postprocessed_plan_length_rel_error_mean")
        online_route = online_task.get("hamiltonian_path_success_mean")
        baseline_route = baseline_task.get("hamiltonian_path_success_mean")
        per_task.append(
            {
                "task_id": int(task_id),
                "online_hamiltonian_success_mean": online_route,
                "baseline_hamiltonian_success_mean": baseline_route,
                "online_postprocessed_rel_error_mean": online_rel,
                "baseline_postprocessed_rel_error_mean": baseline_rel,
                "route_success_delta_online_minus_baseline": (
                    None
                    if online_route is None or baseline_route is None
                    else float(online_route) - float(baseline_route)
                ),
                "rel_error_delta_online_minus_baseline": (
                    None
                    if online_rel is None or baseline_rel is None
                    else float(online_rel) - float(baseline_rel)
                ),
            }
        )

    summary = {
        "dataset": online.get("dataset") or baseline.get("dataset"),
        "run_timestamp": online.get("run_timestamp") or baseline.get("run_timestamp"),
        "online_summary": os.path.abspath(args.online_summary),
        "baseline_summary": os.path.abspath(args.baseline_summary),
        "online_planner_variants": online.get("planner_variants", []),
        "baseline_planner_variants": baseline.get("planner_variants", []),
        "online_final_hamiltonian_success_mean": online.get("final_hamiltonian_success_mean"),
        "baseline_final_hamiltonian_success_mean": baseline.get("final_hamiltonian_success_mean"),
        "online_final_postprocessed_rel_error_mean": online.get("final_postprocessed_rel_error_mean"),
        "baseline_final_postprocessed_rel_error_mean": baseline.get("final_postprocessed_rel_error_mean"),
        "final_route_success_delta_online_minus_baseline": (
            None
            if online.get("final_hamiltonian_success_mean") is None
            or baseline.get("final_hamiltonian_success_mean") is None
            else float(online["final_hamiltonian_success_mean"])
            - float(baseline["final_hamiltonian_success_mean"])
        ),
        "final_rel_error_delta_online_minus_baseline": (
            None
            if online.get("final_postprocessed_rel_error_mean") is None
            or baseline.get("final_postprocessed_rel_error_mean") is None
            else float(online["final_postprocessed_rel_error_mean"])
            - float(baseline["final_postprocessed_rel_error_mean"])
        ),
        "task_comparisons": per_task,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"comparison_summary: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
