import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def load_result_files(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "repeat_*.json")))
    if not files:
        raise FileNotFoundError(f"No benchmark result files found under {results_dir}")

    payloads = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads


def main():
    parser = argparse.ArgumentParser(description="Collect single-checkpoint benchmark results.")
    parser.add_argument("--results_dir", required=True, help="Directory containing repeat/task JSON results")
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--expected_tasks", type=int, default=5)
    parser.add_argument("--expected_rollouts", type=int, default=50)
    args = parser.parse_args()

    payloads = load_result_files(args.results_dir)

    repeat_task_scores = defaultdict(dict)
    model_ids = set()
    for payload in payloads:
        model_id = payload.get("model_id")
        if model_id is not None:
            model_ids.add(model_id)
        repeat_id = payload.get("eval_repeat_id")
        for task_result in payload.get("task_results", []):
            task_id = task_result["task_id"]
            repeat_task_scores[repeat_id][task_id] = float(task_result["task_success_mean"])
            n_rollouts = int(task_result.get("num_rollouts", 0))
            if n_rollouts != args.expected_rollouts:
                print(
                    f"WARNING: repeat {repeat_id} task {task_id} has {n_rollouts} rollouts "
                    f"(expected {args.expected_rollouts})"
                )

    missing = []
    repeat_scores = []
    print("=" * 60)
    print("Single-Checkpoint Benchmark Results")
    print("=" * 60)
    if model_ids:
        print(f"model_id: {sorted(model_ids)}")

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

        if ordered_scores:
            repeat_mean = sum(ordered_scores) / len(ordered_scores)
            repeat_scores.append(repeat_mean)
            print(f"  overall_success: {repeat_mean:.4f}")
        else:
            print("  overall_success: MISSING")

    if repeat_scores:
        final_mean = sum(repeat_scores) / len(repeat_scores)
        if len(repeat_scores) > 1:
            variance = sum((score - final_mean) ** 2 for score in repeat_scores) / len(repeat_scores)
            final_std = math.sqrt(variance)
        else:
            final_std = 0.0
        print("-" * 60)
        print(f"final/repeat_scores: {[round(score, 4) for score in repeat_scores]}")
        print(f"final/overall_success_mean: {final_mean:.4f}")
        print(f"final/overall_success_std: {final_std:.4f}")
    else:
        print("-" * 60)
        print("final/overall_success_mean: MISSING")
        sys.exit(1)

    if missing:
        print(f"WARNING: missing repeat/task pairs: {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
