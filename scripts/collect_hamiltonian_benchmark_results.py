from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from benchmark_summary_utils import load_result_files, mean_or_none, population_std, write_summary

BIN_NAMES = ["high", "mid", "low"]


def _bin_name_for_group(group_idx: int, n_per_bin: int) -> str | None:
    """Map waypoint_group_idx (0-based position in YAML) to a difficulty bin.

    0..n-1 → "high", n..2n-1 → "mid", 2n..3n-1 → "low".
    Indices >= 3n are outside the stratified sample (return None).
    """
    if n_per_bin <= 0:
        return None
    if group_idx < n_per_bin:
        return "high"
    if group_idx < n_per_bin * 2:
        return "mid"
    if group_idx < n_per_bin * 3:
        return "low"
    return None


def _bin_summary(agent_vals, route_vals, rel_vals):
    return {
        "agent_success_mean": mean_or_none(agent_vals),
        "agent_success_std": population_std(agent_vals),
        "hamiltonian_path_success_mean": mean_or_none(route_vals),
        "hamiltonian_path_success_std": population_std(route_vals),
        "postprocessed_plan_length_rel_error_mean": mean_or_none(rel_vals),
        "postprocessed_plan_length_rel_error_std": population_std(rel_vals),
        "n_samples": len(route_vals),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect Hamiltonian benchmark results.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--expected_tasks", type=int, default=5)
    parser.add_argument("--expected_waypoint_top_n", type=int, required=True)
    parser.add_argument("--n_per_bin", type=int, default=0,
                        help="Groups per difficulty bin (0 = derive as expected_waypoint_top_n // 3)")
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--run_timestamp", default=None)
    parser.add_argument("--task_override_snapshot", default=None)
    args = parser.parse_args()

    n_per_bin = args.n_per_bin if args.n_per_bin > 0 else args.expected_waypoint_top_n // 3

    payloads, result_files = load_result_files(args.results_dir, ("*_repeat_*.json",))

    repeat_task_metrics = defaultdict(dict)
    per_task_agent = defaultdict(list)
    per_task_route = defaultdict(list)
    per_task_rel_error = defaultdict(list)
    # (task_id, bin_name) → list of per-group metric values across all repeats
    per_task_bin_agent = defaultdict(lambda: defaultdict(list))
    per_task_bin_route = defaultdict(lambda: defaultdict(list))
    per_task_bin_rel_error = defaultdict(lambda: defaultdict(list))

    warnings = []
    missing = []
    model_ids = set()
    planner_variants = set()
    plan_only_flags = set()

    for payload in payloads:
        model_id = payload.get("model_id")
        if model_id is not None:
            model_ids.add(model_id)
        planner_variant = payload.get("planner_variant")
        if planner_variant is not None:
            planner_variants.add(str(planner_variant))
        plan_only_flags.add(bool(payload.get("plan_only", False)))
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

            agent_mean = task_result.get("task_agent_success_mean")
            agent_mean = float(agent_mean) if agent_mean is not None else None
            route_mean = float(task_result.get("task_hamiltonian_path_success_mean", 0.0))
            rel_error_mean = task_result.get("task_postprocessed_plan_length_rel_error_mean")
            rel_error_mean = float(rel_error_mean) if rel_error_mean is not None else None

            repeat_task_metrics[repeat_id][task_id] = {
                "agent_success_mean": agent_mean,
                "hamiltonian_path_success_mean": route_mean,
                "postprocessed_plan_length_rel_error_mean": rel_error_mean,
                "num_groups": group_count,
            }
            if agent_mean is not None:
                per_task_agent[task_id].append(agent_mean)
            per_task_route[task_id].append(route_mean)
            if rel_error_mean is not None:
                per_task_rel_error[task_id].append(rel_error_mean)

            # Accumulate per-group values for bin-level stats.
            for gr in group_results:
                g_idx = gr.get("waypoint_group_idx")
                if g_idx is None:
                    continue
                bin_name = _bin_name_for_group(int(g_idx), n_per_bin)
                if bin_name is None:
                    continue
                g_agent = gr.get("success")
                g_route = gr.get("hamiltonian_path_success")
                g_relerr = gr.get("postprocessed_plan_length_rel_error")
                if g_agent is not None:
                    per_task_bin_agent[task_id][bin_name].append(float(g_agent))
                if g_route is not None:
                    per_task_bin_route[task_id][bin_name].append(float(g_route))
                if g_relerr is not None:
                    per_task_bin_rel_error[task_id][bin_name].append(float(g_relerr))

    # ── Aggregate ─────────────────────────────────────────────────────────────
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
    if n_per_bin > 0:
        print(f"n_per_bin: {n_per_bin}  bins: {BIN_NAMES}")

    for task_id in range(1, args.expected_tasks + 1):
        task_agent_vals = per_task_agent.get(task_id, [])
        task_route_vals = per_task_route.get(task_id, [])
        task_rel_vals = per_task_rel_error.get(task_id, [])

        bin_summaries = {}
        for bin_name in BIN_NAMES:
            bin_summaries[bin_name] = _bin_summary(
                per_task_bin_agent[task_id][bin_name],
                per_task_bin_route[task_id][bin_name],
                per_task_bin_rel_error[task_id][bin_name],
            )

        task_summaries.append(
            {
                "task_id": task_id,
                "agent_success_mean": mean_or_none(task_agent_vals),
                "agent_success_std_over_repeats": population_std(task_agent_vals),
                "hamiltonian_path_success_mean": mean_or_none(task_route_vals),
                "hamiltonian_path_success_std_over_repeats": population_std(task_route_vals),
                "postprocessed_plan_length_rel_error_mean": mean_or_none(task_rel_vals),
                "postprocessed_plan_length_rel_error_std_over_repeats": population_std(task_rel_vals),
                "repeat_agent_scores": task_agent_vals,
                "repeat_hamiltonian_scores": task_route_vals,
                "repeat_rel_error_scores": task_rel_vals,
                "bin_results": bin_summaries,
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
            agent_text = (
                f"{metrics['agent_success_mean']:.4f}"
                if metrics["agent_success_mean"] is not None
                else "None"
            )
            print(
                f"  task_{task_id}: agent={agent_text} "
                f"route={metrics['hamiltonian_path_success_mean']:.4f} "
                f"relerr={metrics['postprocessed_plan_length_rel_error_mean'] if metrics['postprocessed_plan_length_rel_error_mean'] is not None else 'None'} "
                f"groups={metrics['num_groups']}"
            )
            if metrics["agent_success_mean"] is not None:
                ordered_agent.append(metrics["agent_success_mean"])
            ordered_route.append(metrics["hamiltonian_path_success_mean"])
            if metrics["postprocessed_plan_length_rel_error_mean"] is not None:
                ordered_rel.append(metrics["postprocessed_plan_length_rel_error_mean"])

        agent_repeat_mean = mean_or_none(ordered_agent)
        route_repeat_mean = mean_or_none(ordered_route)
        rel_repeat_mean = mean_or_none(ordered_rel)
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

    # ── Per-task summary ──────────────────────────────────────────────────────
    print("-" * 60)
    for task_summary in task_summaries:
        agent_text = (
            f"{task_summary['agent_success_mean']:.4f}"
            if task_summary["agent_success_mean"] is not None
            else "None"
        )
        rel_text = (
            f"{task_summary['postprocessed_plan_length_rel_error_mean']:.4f}"
            if task_summary["postprocessed_plan_length_rel_error_mean"] is not None
            else "None"
        )
        print(
            f"task_{task_summary['task_id']}: "
            f"agent={agent_text} "
            f"route={task_summary['hamiltonian_path_success_mean']:.4f} "
            f"relerr={rel_text}"
        )

    # ── Per-task × bin summary ────────────────────────────────────────────────
    if n_per_bin > 0:
        print("-" * 60)
        print(f"Per-task × bin breakdown (n_per_bin={n_per_bin}):")
        for task_summary in task_summaries:
            task_id = task_summary["task_id"]
            for bin_name in BIN_NAMES:
                bs = task_summary["bin_results"][bin_name]
                route_m = bs["hamiltonian_path_success_mean"]
                relerr_m = bs["postprocessed_plan_length_rel_error_mean"]
                agent_m = bs["agent_success_mean"]
                print(
                    f"  task_{task_id}/{bin_name}: "
                    f"agent={'%.4f' % agent_m if agent_m is not None else 'None'}  "
                    f"route={'%.4f' % route_m if route_m is not None else 'None'}  "
                    f"relerr={'%.4f' % relerr_m if relerr_m is not None else 'None'}  "
                    f"n={bs['n_samples']}"
                )

    # ── Cross-task bin aggregates ─────────────────────────────────────────────
    bin_cross_task: dict[str, dict] = {}
    if n_per_bin > 0:
        print("-" * 60)
        print("Cross-task bin aggregates:")
        for bin_name in BIN_NAMES:
            all_agent, all_route, all_rel = [], [], []
            for task_id in range(1, args.expected_tasks + 1):
                all_agent.extend(per_task_bin_agent[task_id][bin_name])
                all_route.extend(per_task_bin_route[task_id][bin_name])
                all_rel.extend(per_task_bin_rel_error[task_id][bin_name])
            bs = _bin_summary(all_agent, all_route, all_rel)
            bin_cross_task[bin_name] = bs
            route_m = bs["hamiltonian_path_success_mean"]
            relerr_m = bs["postprocessed_plan_length_rel_error_mean"]
            agent_m = bs["agent_success_mean"]
            print(
                f"  {bin_name}: "
                f"agent={'%.4f' % agent_m if agent_m is not None else 'None'}  "
                f"route={'%.4f' % route_m if route_m is not None else 'None'}  "
                f"relerr={'%.4f' % relerr_m if relerr_m is not None else 'None'}  "
                f"n={bs['n_samples']}"
            )

    # ── Overall finals ────────────────────────────────────────────────────────
    final_agent_mean = mean_or_none(repeat_agent_scores)
    final_route_mean = mean_or_none(repeat_route_scores)
    final_rel_mean = mean_or_none(repeat_rel_error_scores)
    final_agent_std = population_std(repeat_agent_scores)
    final_route_std = population_std(repeat_route_scores)
    final_rel_std = population_std(repeat_rel_error_scores)

    print("-" * 60)
    print(f"final/agent_success_mean: {final_agent_mean if final_agent_mean is not None else 'MISSING'}")
    print(f"final/agent_success_std: {final_agent_std:.4f}")
    print(f"final/hamiltonian_success_mean: {final_route_mean if final_route_mean is not None else 'MISSING'}")
    print(f"final/hamiltonian_success_std: {final_route_std:.4f}")
    print(f"final/postprocessed_rel_error_mean: {final_rel_mean if final_rel_mean is not None else 'MISSING'}")
    print(f"final/postprocessed_rel_error_std: {final_rel_std:.4f}")

    summary = {
        "status": "complete" if not missing and repeat_route_scores else "incomplete",
        "dataset": args.dataset_name,
        "run_timestamp": args.run_timestamp,
        "task_override_snapshot": args.task_override_snapshot,
        "results_dir": os.path.abspath(args.results_dir),
        "result_files": [os.path.abspath(path) for path in result_files],
        "model_ids": sorted(model_ids),
        "planner_variants": sorted(planner_variants),
        "plan_only": all(plan_only_flags) if plan_only_flags else False,
        "expected_repeats": args.expected_repeats,
        "expected_tasks": args.expected_tasks,
        "expected_waypoint_top_n": args.expected_waypoint_top_n,
        "n_per_bin": n_per_bin,
        "repeat_results": repeat_summaries,
        "task_results": task_summaries,
        "bin_cross_task_results": bin_cross_task,
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

    write_summary(args.summary_output, summary)
    print(f"final/summary_json: {os.path.abspath(args.summary_output)}")

    if missing:
        print(f"WARNING: missing repeat/task pairs: {missing}")
        sys.exit(1)

    if final_route_mean is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
