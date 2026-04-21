import argparse
import copy
import json
import os
from datetime import datetime

from project_config import WANDB_ENTITY, WANDB_PROJECT


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-checkpoint benchmark jobs."
    )
    parser.add_argument("--dataset", required=True, help="Hydra dataset config name")
    parser.add_argument("--model_id", required=True, help="Checkpoint model id to load")
    parser.add_argument(
        "--load_path",
        default=None,
        help="Optional explicit checkpoint path/value to pass as Hydra load=...; defaults to model_id",
    )
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of OGBench tasks/goals")
    parser.add_argument("--num_repeats", type=int, default=3, help="Number of repeated evaluation passes")
    parser.add_argument("--rollouts_per_task", type=int, default=50, help="Number of rollouts per (repeat, task)")
    parser.add_argument(
        "--jobs_dir",
        default="jobs",
        help="Directory to write generated benchmark job json files into",
    )
    parser.add_argument(
        "--planning_config_snapshot",
        default=None,
        help="Repo-relative path to the df_planning.yaml snapshot to pin for this benchmark run",
    )
    parser.add_argument("--results_dir", required=True, help="Directory to write benchmark JSON results into")
    parser.add_argument(
        "--results_file_prefix",
        default="benchmark",
        help="Filename prefix for per-task benchmark JSON results",
    )
    args = parser.parse_args()

    os.makedirs(args.jobs_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    basic_job_config = {
        "wandb.entity": WANDB_ENTITY,
        "wandb.project": WANDB_PROJECT,
        "wandb.group": f"BENCH-{args.model_id}",
        "wandb.mode": "online",
        "experiment": "base_pytorch",
        "algorithm": "df_planning",
        "dataset": args.dataset,
        "load": args.load_path or args.model_id,
        "experiment.tasks": ["benchmark"],
        "algorithm.use_anchor_planner": True,
        "algorithm.benchmark_num_rollouts": args.rollouts_per_task,
        "algorithm.benchmark_model_id": args.model_id,
    }
    if args.planning_config_snapshot:
        basic_job_config["+algorithm_snapshot_path"] = args.planning_config_snapshot

    count = 0
    for repeat_id in range(1, args.num_repeats + 1):
        for task_id in range(1, args.num_tasks + 1):
            job_cfg = copy.deepcopy(basic_job_config)
            job_cfg["algorithm.task_id"] = task_id
            job_cfg["algorithm.eval_repeat_id"] = repeat_id
            job_cfg["algorithm.benchmark_rollout_seed_base"] = repeat_id * 100000 + task_id * 1000
            job_cfg["algorithm.benchmark_results_path"] = os.path.join(
                args.results_dir,
                f"{args.results_file_prefix}_repeat_{repeat_id:02d}_task_{task_id:02d}.json",
            )
            job_cfg["+name"] = f"BENCH_{args.model_id}_R{repeat_id}_T{task_id}"

            filename = os.path.join(
                args.jobs_dir,
                f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json",
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(job_cfg, f, indent=2)
            count += 1

    print(f"Successfully generated {count} benchmark jobs in '{args.jobs_dir}/'.")
    print(f"Results will be written under: {args.results_dir}")


if __name__ == "__main__":
    main()
