"""
collect_eval_results.py
-----------------------
Collect per-task success rates from WandB after eval_all.sh jobs finish,
and report them in the same style as ogbench/impls/main.py:

  evaluation/task_1_success : 0.XX
  evaluation/task_2_success : 0.XX
  ...
  evaluation/overall_success: 0.XX
"""

import argparse
import sys
import time
from collections import defaultdict

import yaml
from project_config import WANDB_ENTITY
WANDB_PROJECT = "mctd_eval"

# Possible metric keys logged by df_planning.py's interact()
SUCCESS_METRIC_KEYS = [
    "validation/success_rate",
    "validation_no_guidance_random_walk/success_rate",
]


def get_num_tasks_from_yaml(dataset: str) -> int:
    yaml_path = f"configurations/dataset/{dataset}.yaml"
    try:
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
        return int(d.get("num_tasks", 5))
    except Exception:
        return 5


def get_task_names(env_id: str, num_tasks: int):
    """Try to get task names from ogbench environment; fall back to task_N."""
    try:
        import sys, os
        ogbench_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "ogbench", "impls")
        )
        if ogbench_path not in sys.path:
            sys.path.insert(0, ogbench_path)
        from utils.env_utils import make_env_and_datasets
        env, _, _ = make_env_and_datasets(env_id, frame_stack=1)
        task_infos = (
            env.unwrapped.task_infos
            if hasattr(env.unwrapped, "task_infos")
            else env.task_infos
        )
        names = {i + 1: task_infos[i]["task_name"] for i in range(len(task_infos))}
        env.close()
        return names
    except Exception:
        return {i: f"task_{i}" for i in range(1, num_tasks + 1)}


def collect_from_wandb(model_id: str, num_tasks: int, max_wait: int = 120):
    """
    Query WandB for all runs in group EVAL-{model_id}.
    Returns dict: task_id -> list of success_rate values (one per seed run).
    Waits up to max_wait seconds for runs to appear.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb package not found. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api(timeout=60)
    group = f"EVAL-{model_id}"

    print(f"Querying WandB: entity={WANDB_ENTITY}, project={WANDB_PROJECT}, group={group}")

    runs = []
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            runs = list(api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"group": group, "state": "finished"},
            ))
        except Exception as e:
            print(f"  WandB query error: {e}")
            time.sleep(5)
            continue

        if len(runs) > 0:
            print(f"  Found {len(runs)} finished run(s) in group {group}")
            break
        else:
            remaining = int(deadline - time.time())
            print(f"  No finished runs yet. Waiting... ({remaining}s remaining)")
            time.sleep(10)

    if not runs:
        print(f"WARNING: No finished runs found for group {group} after {max_wait}s.")
        print(f"  Check WandB dashboard: {WANDB_PROJECT} / group={group}")
        return {}

    task_results = defaultdict(list)

    for run in runs:
        # Get task_id from run config
        cfg = run.config
        task_id = (
            cfg.get("algorithm.task_id")
            or cfg.get("algorithm", {}).get("task_id")
        )
        if task_id is None:
            print(f"  WARNING: run {run.name} has no task_id in config, skipping.")
            continue
        task_id = int(task_id)

        # Get success rate from summary (last logged value)
        summary = run.summary
        success = None
        for key in SUCCESS_METRIC_KEYS:
            val = summary.get(key)
            if val is not None:
                success = float(val)
                break

        if success is None:
            print(f"  WARNING: run {run.name} (task {task_id}) has no success metric, skipping.")
            print(f"    Available summary keys: {list(summary.keys())[:10]}")
            continue

        task_results[task_id].append(success)
        print(f"  run {run.name}: task_id={task_id}, success={success:.4f}")

    return dict(task_results)


def report_results(task_results: dict, task_names: dict, num_tasks: int):
    """
    Print per-task and overall success rates, matching ogbench/impls/main.py output style.

      evaluation/{task_name}_success : X.XXXX
      ...
      evaluation/overall_success     : X.XXXX
    """
    print("")
    print("=" * 55)
    print("  Evaluation Results (OGBench-style)")
    print("=" * 55)

    per_task_means = {}
    missing_tasks = []

    for task_id in range(1, num_tasks + 1):
        name = task_names.get(task_id, f"task_{task_id}")
        values = task_results.get(task_id, [])
        if values:
            mean_val = sum(values) / len(values)
            per_task_means[task_id] = mean_val
            seeds_str = ", ".join(f"{v:.4f}" for v in values)
            print(f"  evaluation/{name}_success : {mean_val:.4f}  (seeds: [{seeds_str}])")
        else:
            missing_tasks.append(task_id)
            print(f"  evaluation/{name}_success : N/A  (no data for task {task_id})")

    if per_task_means:
        overall = sum(per_task_means.values()) / len(per_task_means)
        print("-" * 55)
        print(f"  evaluation/overall_success     : {overall:.4f}")
        print(f"  (averaged over {len(per_task_means)}/{num_tasks} tasks)")
    else:
        print("-" * 55)
        print("  evaluation/overall_success     : N/A (no results collected)")

    if missing_tasks:
        print(f"\n  WARNING: Missing results for task IDs: {missing_tasks}")

    print("=" * 55)
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and report success rates from WandB after eval_all.sh."
    )
    parser.add_argument("--model_id",  required=True, help="WandB model ID (e.g., en1ddvu7)")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Number of tasks (overrides dataset yaml)")
    parser.add_argument("--dataset",   default=None,
                        help="Dataset config name (used to read num_tasks and env_id)")
    parser.add_argument("--max_wait",  type=int, default=120,
                        help="Max seconds to wait for WandB runs to appear (default: 120)")
    args = parser.parse_args()

    # Determine num_tasks
    if args.num_tasks is not None:
        num_tasks = args.num_tasks
    elif args.dataset is not None:
        num_tasks = get_num_tasks_from_yaml(args.dataset)
    else:
        num_tasks = 5
        print(f"WARNING: --num_tasks and --dataset not given, defaulting to {num_tasks}")

    # Try to get proper task names from ogbench environment
    task_names = {}
    if args.dataset is not None:
        try:
            env_id_yaml = f"configurations/dataset/{args.dataset}.yaml"
            with open(env_id_yaml) as f:
                ds_cfg = yaml.safe_load(f)
            env_id = ds_cfg.get("env_id")
            if env_id:
                task_names = get_task_names(env_id, num_tasks)
        except Exception:
            pass
    if not task_names:
        task_names = {i: f"task_{i}" for i in range(1, num_tasks + 1)}

    # Collect results from WandB
    task_results = collect_from_wandb(args.model_id, num_tasks, max_wait=args.max_wait)

    # Report
    report_results(task_results, task_names, num_tasks)

    # Exit 1 if no results at all
    if not task_results:
        sys.exit(1)


if __name__ == "__main__":
    main()
