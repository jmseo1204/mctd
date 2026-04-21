from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

from benchmark_summary_utils import load_result_files, mean_or_none, population_std, write_summary


def _extract_optional_mean(
    task_result: dict,
    rollouts: list[dict],
    *,
    task_key: str,
    rollout_key: str,
    finite_only: bool = False,
) -> float | None:
    value = task_result.get(task_key)
    if value is not None:
        value = float(value)
        if finite_only and not math.isfinite(value):
            return None
        return value

    values = []
    for rollout in rollouts:
        rollout_value = rollout.get(rollout_key)
        if rollout_value is None:
            continue
        rollout_value = float(rollout_value)
        if finite_only and not math.isfinite(rollout_value):
            continue
        values.append(rollout_value)
    return mean_or_none(values)


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

    payloads, result_files = load_result_files(args.results_dir, ("*_repeat_*.json", "repeat_*.json"))

    repeat_task_metrics = defaultdict(dict)
    task_rollout_successes = defaultdict(list)
    repeat_rollout_successes = defaultdict(list)
    per_task_success = defaultdict(list)
    per_task_postprocessed_plan_length = defaultdict(list)
    per_task_groundtruth_path_length = defaultdict(list)
    per_task_rel_error = defaultdict(list)
    warnings = []
    model_ids = set()

    for payload in payloads:
        model_id = payload.get("model_id")
        if model_id is not None:
            model_ids.add(model_id)
        repeat_id = int(payload.get("eval_repeat_id"))
        for task_result in payload.get("task_results", []):
            task_id = int(task_result["task_id"])
            score = float(task_result["task_success_mean"])
            rollouts = task_result.get("rollouts", [])
            post_mean = _extract_optional_mean(
                task_result,
                rollouts,
                task_key="task_postprocessed_plan_length_mean",
                rollout_key="postprocessed_plan_length",
            )
            groundtruth_mean = _extract_optional_mean(
                task_result,
                rollouts,
                task_key="task_groundtruth_path_length_mean",
                rollout_key="groundtruth_path_length",
                finite_only=True,
            )
            rel_error_mean = _extract_optional_mean(
                task_result,
                rollouts,
                task_key="task_postprocessed_plan_length_rel_error_mean",
                rollout_key="postprocessed_plan_length_rel_error",
                finite_only=True,
            )

            repeat_task_metrics[repeat_id][task_id] = {
                "success_mean": score,
                "postprocessed_plan_length_mean": post_mean,
                "groundtruth_path_length_mean": groundtruth_mean,
                "postprocessed_plan_length_rel_error_mean": rel_error_mean,
                "num_rollouts": len(rollouts) if rollouts else int(task_result.get("num_rollouts", 0)),
            }
            per_task_success[task_id].append(score)
            if post_mean is not None:
                per_task_postprocessed_plan_length[task_id].append(post_mean)
            if groundtruth_mean is not None:
                per_task_groundtruth_path_length[task_id].append(groundtruth_mean)
            if rel_error_mean is not None:
                per_task_rel_error[task_id].append(rel_error_mean)

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
    repeat_postprocessed_plan_length_scores = []
    repeat_groundtruth_path_length_scores = []
    repeat_rel_error_scores = []
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
            task_std = population_std(per_task_success.get(task_id, []))
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
                "postprocessed_plan_length_mean": mean_or_none(
                    per_task_postprocessed_plan_length.get(task_id, [])
                ),
                "postprocessed_plan_length_std_over_repeats": population_std(
                    per_task_postprocessed_plan_length.get(task_id, [])
                ),
                "groundtruth_path_length_mean": mean_or_none(
                    per_task_groundtruth_path_length.get(task_id, [])
                ),
                "groundtruth_path_length_std_over_repeats": population_std(
                    per_task_groundtruth_path_length.get(task_id, [])
                ),
                "postprocessed_plan_length_rel_error_mean": mean_or_none(
                    per_task_rel_error.get(task_id, [])
                ),
                "postprocessed_plan_length_rel_error_std_over_repeats": population_std(
                    per_task_rel_error.get(task_id, [])
                ),
                "repeat_scores": per_task_success.get(task_id, []),
                "repeat_postprocessed_plan_length_scores": per_task_postprocessed_plan_length.get(
                    task_id, []
                ),
                "repeat_groundtruth_path_length_scores": per_task_groundtruth_path_length.get(
                    task_id, []
                ),
                "repeat_rel_error_scores": per_task_rel_error.get(task_id, []),
            }
        )

    for repeat_id in range(1, args.expected_repeats + 1):
        task_metrics = repeat_task_metrics.get(repeat_id, {})
        print(f"repeat_{repeat_id}:")
        ordered_scores = []
        ordered_post_lengths = []
        ordered_groundtruth_lengths = []
        ordered_rel_errors = []
        for task_id in range(1, args.expected_tasks + 1):
            metrics = task_metrics.get(task_id)
            if metrics is None:
                missing.append((repeat_id, task_id))
                print(f"  task_{task_id}: MISSING")
                continue
            print(
                f"  task_{task_id}: success={metrics['success_mean']:.4f} "
                f"post_len={metrics['postprocessed_plan_length_mean'] if metrics['postprocessed_plan_length_mean'] is not None else 'None'} "
                f"gt_len={metrics['groundtruth_path_length_mean'] if metrics['groundtruth_path_length_mean'] is not None else 'None'} "
                f"relerr={metrics['postprocessed_plan_length_rel_error_mean'] if metrics['postprocessed_plan_length_rel_error_mean'] is not None else 'None'} "
                f"rollouts={metrics['num_rollouts']}"
            )
            ordered_scores.append(metrics["success_mean"])
            if metrics["postprocessed_plan_length_mean"] is not None:
                ordered_post_lengths.append(metrics["postprocessed_plan_length_mean"])
            if metrics["groundtruth_path_length_mean"] is not None:
                ordered_groundtruth_lengths.append(metrics["groundtruth_path_length_mean"])
            if metrics["postprocessed_plan_length_rel_error_mean"] is not None:
                ordered_rel_errors.append(metrics["postprocessed_plan_length_rel_error_mean"])

        successes = repeat_rollout_successes.get(repeat_id, [])
        success_count = int(sum(successes))
        num_rollouts = len(successes)
        success_mean = (success_count / num_rollouts) if num_rollouts else float("nan")
        post_length_mean = mean_or_none(ordered_post_lengths)
        groundtruth_length_mean = mean_or_none(ordered_groundtruth_lengths)
        rel_error_mean = mean_or_none(ordered_rel_errors)
        repeat_summaries.append(
            {
                "repeat_id": repeat_id,
                "success_count": success_count,
                "num_rollouts": num_rollouts,
                "success_mean": success_mean,
                "postprocessed_plan_length_mean": post_length_mean,
                "groundtruth_path_length_mean": groundtruth_length_mean,
                "postprocessed_plan_length_rel_error_mean": rel_error_mean,
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
        if post_length_mean is not None:
            repeat_postprocessed_plan_length_scores.append(post_length_mean)
            print(f"  overall_postprocessed_plan_length: {post_length_mean:.4f}")
        else:
            print("  overall_postprocessed_plan_length: MISSING")
        if groundtruth_length_mean is not None:
            repeat_groundtruth_path_length_scores.append(groundtruth_length_mean)
            print(f"  overall_groundtruth_path_length: {groundtruth_length_mean:.4f}")
        else:
            print("  overall_groundtruth_path_length: MISSING")
        if rel_error_mean is not None:
            repeat_rel_error_scores.append(rel_error_mean)
            print(f"  overall_rel_error: {rel_error_mean:.4f}")
        else:
            print("  overall_rel_error: MISSING")

    total_rollout_successes = []
    for task_id in range(1, args.expected_tasks + 1):
        total_rollout_successes.extend(task_rollout_successes.get(task_id, []))

    total_success_count = int(sum(total_rollout_successes))
    total_num_rollouts = len(total_rollout_successes)
    total_success_mean = (
        total_success_count / total_num_rollouts if total_num_rollouts else float("nan")
    )
    total_success_std = population_std(repeat_scores)
    final_postprocessed_plan_length_mean = mean_or_none(repeat_postprocessed_plan_length_scores)
    final_postprocessed_plan_length_std = population_std(repeat_postprocessed_plan_length_scores)
    final_groundtruth_path_length_mean = mean_or_none(repeat_groundtruth_path_length_scores)
    final_groundtruth_path_length_std = population_std(repeat_groundtruth_path_length_scores)
    final_rel_error_mean = mean_or_none(repeat_rel_error_scores)
    final_rel_error_std = population_std(repeat_rel_error_scores)

    print("-" * 60)
    for task_summary in task_summaries:
        if task_summary["num_rollouts"] == 0:
            print(f"task_{task_summary['task_id']}: success_count=MISSING mean=MISSING")
            continue
        print(
            f"task_{task_summary['task_id']}: "
            f"success_count={task_summary['success_count']}/{task_summary['num_rollouts']} "
            f"mean={task_summary['success_mean']:.4f} "
            f"post_len={task_summary['postprocessed_plan_length_mean'] if task_summary['postprocessed_plan_length_mean'] is not None else 'None'} "
            f"gt_len={task_summary['groundtruth_path_length_mean'] if task_summary['groundtruth_path_length_mean'] is not None else 'None'} "
            f"relerr={task_summary['postprocessed_plan_length_rel_error_mean'] if task_summary['postprocessed_plan_length_rel_error_mean'] is not None else 'None'}"
        )

    if repeat_scores:
        print("-" * 60)
        print(f"final/repeat_scores: {[round(score, 4) for score in repeat_scores]}")
        print(f"final/total_success_count: {total_success_count}/{total_num_rollouts}")
        print(f"final/overall_success_mean: {total_success_mean:.4f}")
        print(f"final/overall_success_std: {total_success_std:.4f}")
        print(
            "final/postprocessed_plan_length_mean: "
            f"{final_postprocessed_plan_length_mean if final_postprocessed_plan_length_mean is not None else 'MISSING'}"
        )
        print(f"final/postprocessed_plan_length_std: {final_postprocessed_plan_length_std:.4f}")
        print(
            "final/groundtruth_path_length_mean: "
            f"{final_groundtruth_path_length_mean if final_groundtruth_path_length_mean is not None else 'MISSING'}"
        )
        print(f"final/groundtruth_path_length_std: {final_groundtruth_path_length_std:.4f}")
        print(
            "final/postprocessed_plan_rel_error_mean: "
            f"{final_rel_error_mean if final_rel_error_mean is not None else 'MISSING'}"
        )
        print(f"final/postprocessed_plan_rel_error_std: {final_rel_error_std:.4f}")
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
        "repeat_postprocessed_plan_length_scores": repeat_postprocessed_plan_length_scores,
        "repeat_groundtruth_path_length_scores": repeat_groundtruth_path_length_scores,
        "repeat_postprocessed_plan_length_rel_error_scores": repeat_rel_error_scores,
        "total_success_count": total_success_count,
        "total_num_rollouts": total_num_rollouts,
        "total_success_mean": total_success_mean,
        "total_success_std": total_success_std,
        "total_success_std_over_repeats": total_success_std,
        "final_postprocessed_plan_length_mean": final_postprocessed_plan_length_mean,
        "final_postprocessed_plan_length_std": final_postprocessed_plan_length_std,
        "final_groundtruth_path_length_mean": final_groundtruth_path_length_mean,
        "final_groundtruth_path_length_std": final_groundtruth_path_length_std,
        "final_postprocessed_plan_length_rel_error_mean": final_rel_error_mean,
        "final_postprocessed_plan_length_rel_error_std": final_rel_error_std,
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
