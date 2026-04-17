import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def load_result_files(results_dir: str):
    patterns = ("*_repeat_*.json", "repeat_*.json")
    files = []
    for pattern in patterns:
        files = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No benchmark result files found under {results_dir}")

    payloads = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads, files


def population_std(values):
    if not values:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((value - mean_val) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def write_summary(summary_output: str, summary: dict):
    summary_dir = os.path.dirname(summary_output)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect single-checkpoint benchmark results.")
    parser.add_argument("--results_dir", required=True, help="Directory containing repeat/task JSON results")
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--expected_tasks", type=int, default=5)
    parser.add_argument("--expected_rollouts", type=int, default=50)
    parser.add_argument("--summary_output", default=None, help="Optional JSON file to write aggregate results")
    parser.add_argument("--dataset_name", default=None, help="Dataset name for the saved summary metadata")
    parser.add_argument("--run_timestamp", default=None, help="Run timestamp for the saved summary metadata")
    args = parser.parse_args()

    payloads, result_files = load_result_files(args.results_dir)

    repeat_task_scores = defaultdict(dict)
    task_rollout_successes = defaultdict(list)
    repeat_rollout_successes = defaultdict(list)
    task_repeat_scores = defaultdict(list)
    warnings = []
    model_ids = set()

    for payload in payloads:
        model_id = payload.get("model_id")
        if model_id is not None:
            model_ids.add(model_id)
        repeat_id = payload.get("eval_repeat_id")
        for task_result in payload.get("task_results", []):
            task_id = int(task_result["task_id"])
            score = float(task_result["task_success_mean"])
            repeat_task_scores[repeat_id][task_id] = score
            task_repeat_scores[task_id].append(score)

            rollouts = task_result.get("rollouts", [])
            if rollouts:
                rollout_successes = [int(bool(rollout.get("success", False))) for rollout in rollouts]
            else:
                n_rollouts = int(task_result.get("num_rollouts", 0))
                rollout_successes = [1] * int(round(score * n_rollouts)) + [0] * max(
                    n_rollouts - int(round(score * n_rollouts)), 0
                )

            n_rollouts = len(rollout_successes)
            if n_rollouts != args.expected_rollouts:
                warning = (
                    f"repeat {repeat_id} task {task_id} has {n_rollouts} rollouts "
                    f"(expected {args.expected_rollouts})"
                )
                warnings.append(warning)
                print(f"WARNING: {warning}")

            task_rollout_successes[task_id].extend(rollout_successes)
            repeat_rollout_successes[repeat_id].extend(rollout_successes)

    missing = []
    repeat_scores = []
    repeat_summaries = []
    task_summaries = []

    print("=" * 60)
    print("Single-Checkpoint Benchmark Results")
    print("=" * 60)
    if model_ids:
        print(f"model_id: {sorted(model_ids)}")

    for task_id in range(1, args.expected_tasks + 1):
        successes = task_rollout_successes.get(task_id, [])
        if successes:
            success_count = int(sum(successes))
            num_rollouts = len(successes)
            success_mean = success_count / num_rollouts
            task_std = population_std(task_repeat_scores.get(task_id, []))
        else:
            success_count = 0
            num_rollouts = 0
            success_mean = float("nan")
            task_std = 0.0
        task_summaries.append(
            {
                "task_id": task_id,
                "success_count": success_count,
                "num_rollouts": num_rollouts,
                "success_mean": success_mean,
                "success_std_over_repeats": task_std,
                "repeat_scores": task_repeat_scores.get(task_id, []),
            }
        )

    for repeat_id in range(1, args.expected_repeats + 1):
        task_scores = repeat_task_scores.get(repeat_id, {})
        print(f"repeat_{repeat_id}:")
        ordered_scores = []
        for task_id in range(1, args.expected_tasks + 1):
            score = task_scores.get(task_id)
            if score is None:
                missing.append((repeat_id, task_id))
                print(f"  task_{task_id}_success: MISSING")
                continue
            print(f"  task_{task_id}_success: {score:.4f}")
            ordered_scores.append(score)

        successes = repeat_rollout_successes.get(repeat_id, [])
        success_count = int(sum(successes))
        num_rollouts = len(successes)
        success_mean = (success_count / num_rollouts) if num_rollouts else float("nan")
        repeat_summaries.append(
            {
                "repeat_id": repeat_id,
                "success_count": success_count,
                "num_rollouts": num_rollouts,
                "success_mean": success_mean,
            }
        )

        if ordered_scores:
            repeat_mean = sum(ordered_scores) / len(ordered_scores)
            repeat_scores.append(repeat_mean)
            print(f"  overall_success: {repeat_mean:.4f}")
            print(f"  success_count : {success_count}/{num_rollouts}")
        else:
            print("  overall_success: MISSING")
            print("  success_count : MISSING")

    total_rollout_successes = []
    for task_id in range(1, args.expected_tasks + 1):
        total_rollout_successes.extend(task_rollout_successes.get(task_id, []))

    total_success_count = int(sum(total_rollout_successes))
    total_num_rollouts = len(total_rollout_successes)
    total_success_mean = (
        total_success_count / total_num_rollouts if total_num_rollouts else float("nan")
    )
    total_success_std = population_std(repeat_scores)

    print("-" * 60)
    for task_summary in task_summaries:
        if task_summary["num_rollouts"] == 0:
            print(f"task_{task_summary['task_id']}: success_count=MISSING mean=MISSING")
            continue
        print(
            f"task_{task_summary['task_id']}: "
            f"success_count={task_summary['success_count']}/{task_summary['num_rollouts']} "
            f"mean={task_summary['success_mean']:.4f}"
        )

    if repeat_scores:
        print("-" * 60)
        print(f"final/repeat_scores: {[round(score, 4) for score in repeat_scores]}")
        print(f"final/total_success_count: {total_success_count}/{total_num_rollouts}")
        print(f"final/overall_success_mean: {total_success_mean:.4f}")
        print(f"final/overall_success_std: {total_success_std:.4f}")
    else:
        print("final/total_success_count: MISSING")
        print("final/overall_success_mean: MISSING")

    summary = {
        "status": "complete" if not missing and repeat_scores else "incomplete",
        "dataset": args.dataset_name,
        "run_timestamp": args.run_timestamp,
        "results_dir": os.path.abspath(args.results_dir),
        "result_files": [os.path.abspath(path) for path in result_files],
        "model_ids": sorted(model_ids),
        "expected_repeats": args.expected_repeats,
        "expected_tasks": args.expected_tasks,
        "expected_rollouts_per_task": args.expected_rollouts,
        "repeat_results": repeat_summaries,
        "task_results": task_summaries,
        "repeat_scores": repeat_scores,
        "total_success_count": total_success_count,
        "total_num_rollouts": total_num_rollouts,
        "total_success_mean": total_success_mean,
        "total_success_std": total_success_std,
        "total_success_std_over_repeats": total_success_std,
        "missing_pairs": [
            {"repeat_id": repeat_id, "task_id": task_id} for repeat_id, task_id in missing
        ],
        "warnings": warnings,
    }

    if args.summary_output:
        write_summary(args.summary_output, summary)
        print(f"final/summary_json: {os.path.abspath(args.summary_output)}")

    if not repeat_scores:
        sys.exit(1)

    if missing:
        print(f"WARNING: missing repeat/task pairs: {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
