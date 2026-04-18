import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def _load_result_files(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "*_repeat_*.json")))
    if not files:
        raise FileNotFoundError(f"No Hamiltonian benchmark result files found under {results_dir}")

    payloads = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads, files


def _population_std(values):
    if not values:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((value - mean_val) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def _write_summary(summary_output: str, summary: dict):
    summary_dir = os.path.dirname(summary_output)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect Hamiltonian benchmark results.")
    parser.add_argument("--results_dir", required=True, help="Directory containing repeat/task JSON results")
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--expected_tasks", type=int, default=5)
    parser.add_argument("--expected_waypoint_top_n", type=int, required=True)
    parser.add_argument("--summary_output", required=True, help="JSON file to write aggregate results")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--run_timestamp", default=None)
    parser.add_argument("--task_override_snapshot", default=None)
    args = parser.parse_args()

    payloads, result_files = _load_result_files(args.results_dir)

    repeat_task_metrics = defaultdict(dict)
    per_task_agent = defaultdict(list)
    per_task_route = defaultdict(list)
    per_task_rel_error = defaultdict(list)
    warnings = []
    missing = []
    model_ids = set()

    for payload in payloads:
        model_id = payload.get("model_id")
        if model_id is not None:
            model_ids.add(model_id)
        repeat_id = int(payload.get("eval_repeat_id"))
        for task_result in payload.get("task_results", []):
            task_id = int(task_result["task_id"])
            group_results = task_result.get("group_results", [])
            group_count = len(group_results)
            if group_count != args.expected_waypoint_top_n:
                warning = (
                    f"repeat {repeat_id} task {task_id} evaluated {group_count} waypoint groups "
                    f"(expected {args.expected_waypoint_top_n})"
                )
                warnings.append(warning)
                print(f"WARNING: {warning}")

            agent_mean = float(task_result.get("task_agent_success_mean", 0.0))
            route_mean = float(task_result.get("task_hamiltonian_path_success_mean", 0.0))
            rel_error_mean = task_result.get("task_postprocessed_plan_length_rel_error_mean")
            rel_error_mean = float(rel_error_mean) if rel_error_mean is not None else None

            repeat_task_metrics[repeat_id][task_id] = {
                "agent_success_mean": agent_mean,
                "hamiltonian_path_success_mean": route_mean,
                "postprocessed_plan_length_rel_error_mean": rel_error_mean,
                "num_groups": group_count,
            }
            per_task_agent[task_id].append(agent_mean)
            per_task_route[task_id].append(route_mean)
            if rel_error_mean is not None:
                per_task_rel_error[task_id].append(rel_error_mean)

    repeat_summaries = []
    task_summaries = []
    repeat_agent_scores = []
    repeat_route_scores = []
    repeat_rel_error_scores = []

    print("=" * 60)
    print("Hamiltonian Benchmark Results")
    print("=" * 60)
    if model_ids:
        print(f"model_id: {sorted(model_ids)}")

    for task_id in range(1, args.expected_tasks + 1):
        task_agent_vals = per_task_agent.get(task_id, [])
        task_route_vals = per_task_route.get(task_id, [])
        task_rel_vals = per_task_rel_error.get(task_id, [])
        task_summaries.append(
            {
                "task_id": task_id,
                "agent_success_mean": _mean_or_none(task_agent_vals),
                "agent_success_std_over_repeats": _population_std(task_agent_vals),
                "hamiltonian_path_success_mean": _mean_or_none(task_route_vals),
                "hamiltonian_path_success_std_over_repeats": _population_std(task_route_vals),
                "postprocessed_plan_length_rel_error_mean": _mean_or_none(task_rel_vals),
                "postprocessed_plan_length_rel_error_std_over_repeats": _population_std(task_rel_vals),
                "repeat_agent_scores": task_agent_vals,
                "repeat_hamiltonian_scores": task_route_vals,
                "repeat_rel_error_scores": task_rel_vals,
            }
        )

    for repeat_id in range(1, args.expected_repeats + 1):
        task_metrics = repeat_task_metrics.get(repeat_id, {})
        print(f"repeat_{repeat_id}:")
        ordered_agent = []
        ordered_route = []
        ordered_rel = []
        for task_id in range(1, args.expected_tasks + 1):
            metrics = task_metrics.get(task_id)
            if metrics is None:
                missing.append((repeat_id, task_id))
                print(f"  task_{task_id}: MISSING")
                continue
            print(
                f"  task_{task_id}: agent={metrics['agent_success_mean']:.4f} "
                f"route={metrics['hamiltonian_path_success_mean']:.4f} "
                f"relerr={metrics['postprocessed_plan_length_rel_error_mean'] if metrics['postprocessed_plan_length_rel_error_mean'] is not None else 'None'} "
                f"groups={metrics['num_groups']}"
            )
            ordered_agent.append(metrics["agent_success_mean"])
            ordered_route.append(metrics["hamiltonian_path_success_mean"])
            if metrics["postprocessed_plan_length_rel_error_mean"] is not None:
                ordered_rel.append(metrics["postprocessed_plan_length_rel_error_mean"])

        agent_repeat_mean = _mean_or_none(ordered_agent)
        route_repeat_mean = _mean_or_none(ordered_route)
        rel_repeat_mean = _mean_or_none(ordered_rel)
        repeat_summaries.append(
            {
                "repeat_id": repeat_id,
                "agent_success_mean": agent_repeat_mean,
                "hamiltonian_path_success_mean": route_repeat_mean,
                "postprocessed_plan_length_rel_error_mean": rel_repeat_mean,
            }
        )
        if agent_repeat_mean is not None:
            repeat_agent_scores.append(agent_repeat_mean)
            print(f"  overall_agent_success: {agent_repeat_mean:.4f}")
        else:
            print("  overall_agent_success: MISSING")
        if route_repeat_mean is not None:
            repeat_route_scores.append(route_repeat_mean)
            print(f"  overall_hamiltonian_success: {route_repeat_mean:.4f}")
        else:
            print("  overall_hamiltonian_success: MISSING")
        if rel_repeat_mean is not None:
            repeat_rel_error_scores.append(rel_repeat_mean)
            print(f"  overall_rel_error: {rel_repeat_mean:.4f}")
        else:
            print("  overall_rel_error: MISSING")

    print("-" * 60)
    for task_summary in task_summaries:
        if task_summary["agent_success_mean"] is None:
            print(f"task_{task_summary['task_id']}: MISSING")
            continue
        rel_text = (
            f"{task_summary['postprocessed_plan_length_rel_error_mean']:.4f}"
            if task_summary["postprocessed_plan_length_rel_error_mean"] is not None
            else "None"
        )
        print(
            f"task_{task_summary['task_id']}: "
            f"agent={task_summary['agent_success_mean']:.4f} "
            f"route={task_summary['hamiltonian_path_success_mean']:.4f} "
            f"relerr={rel_text}"
        )

    final_agent_mean = _mean_or_none(repeat_agent_scores)
    final_route_mean = _mean_or_none(repeat_route_scores)
    final_rel_mean = _mean_or_none(repeat_rel_error_scores)
    final_agent_std = _population_std(repeat_agent_scores)
    final_route_std = _population_std(repeat_route_scores)
    final_rel_std = _population_std(repeat_rel_error_scores)

    print("-" * 60)
    print(f"final/agent_success_mean: {final_agent_mean if final_agent_mean is not None else 'MISSING'}")
    print(f"final/agent_success_std: {final_agent_std:.4f}")
    print(f"final/hamiltonian_success_mean: {final_route_mean if final_route_mean is not None else 'MISSING'}")
    print(f"final/hamiltonian_success_std: {final_route_std:.4f}")
    print(f"final/postprocessed_rel_error_mean: {final_rel_mean if final_rel_mean is not None else 'MISSING'}")
    print(f"final/postprocessed_rel_error_std: {final_rel_std:.4f}")

    summary = {
        "status": "complete" if not missing and repeat_agent_scores and repeat_route_scores else "incomplete",
        "dataset": args.dataset_name,
        "run_timestamp": args.run_timestamp,
        "task_override_snapshot": args.task_override_snapshot,
        "results_dir": os.path.abspath(args.results_dir),
        "result_files": [os.path.abspath(path) for path in result_files],
        "model_ids": sorted(model_ids),
        "expected_repeats": args.expected_repeats,
        "expected_tasks": args.expected_tasks,
        "expected_waypoint_top_n": args.expected_waypoint_top_n,
        "repeat_results": repeat_summaries,
        "task_results": task_summaries,
        "repeat_agent_scores": repeat_agent_scores,
        "repeat_hamiltonian_scores": repeat_route_scores,
        "repeat_postprocessed_rel_error_scores": repeat_rel_error_scores,
        "final_agent_success_mean": final_agent_mean,
        "final_agent_success_std": final_agent_std,
        "final_hamiltonian_success_mean": final_route_mean,
        "final_hamiltonian_success_std": final_route_std,
        "final_postprocessed_rel_error_mean": final_rel_mean,
        "final_postprocessed_rel_error_std": final_rel_std,
        "missing_pairs": [
            {"repeat_id": repeat_id, "task_id": task_id} for repeat_id, task_id in missing
        ],
        "warnings": warnings,
    }

    _write_summary(args.summary_output, summary)
    print(f"final/summary_json: {os.path.abspath(args.summary_output)}")

    if missing:
        print(f"WARNING: missing repeat/task pairs: {missing}")
        sys.exit(1)

    if final_agent_mean is None or final_route_mean is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
