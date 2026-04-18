import argparse
import copy
import json
import os
from datetime import datetime

from project_config import WANDB_ENTITY, WANDB_PROJECT


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hamiltonian benchmark jobs for a single checkpoint."
    )
    parser.add_argument("--dataset", required=True, help="Hydra dataset config name")
    parser.add_argument("--model_id", required=True, help="Checkpoint model id to load")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of OGBench tasks/goals")
    parser.add_argument("--num_repeats", type=int, default=3, help="Number of repeated evaluation passes")
    parser.add_argument(
        "--waypoint_top_n",
        type=int,
        required=True,
        help="Evaluate the first N ranked waypoint groups per task",
    )
    parser.add_argument(
        "--task_override_path",
        required=True,
        help="Repo-relative task override YAML/JSON path to snapshot and evaluate",
    )
    parser.add_argument(
        "--planning_config_snapshot",
        default=None,
        help="Repo-relative path to the df_planning.yaml snapshot to pin for this benchmark run",
    )
    parser.add_argument("--results_dir", required=True, help="Directory to write benchmark JSON results into")
    parser.add_argument(
        "--results_file_prefix",
        default="hamiltonian_benchmark",
        help="Filename prefix for per-task benchmark JSON results",
    )
    args = parser.parse_args()

    if args.waypoint_top_n <= 0:
        raise ValueError(f"--waypoint_top_n must be positive, got {args.waypoint_top_n}")

    os.makedirs("jobs", exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    basic_job_config = {
        "wandb.entity": WANDB_ENTITY,
        "wandb.project": WANDB_PROJECT,
        "wandb.group": f"HBENCH-{args.model_id}",
        "wandb.mode": "online",
        "experiment": "base_pytorch",
        "algorithm": "df_planning",
        "dataset": args.dataset,
        "load": args.model_id,
        "experiment.tasks": ["benchmark"],
        "algorithm.multi_tree_hemiltonian": True,
        "algorithm.task_override_path": args.task_override_path,
        "algorithm.benchmark_waypoint_top_n": int(args.waypoint_top_n),
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
            job_cfg["+name"] = f"HBENCH_{args.model_id}_R{repeat_id}_T{task_id}"

            filename = f"jobs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(job_cfg, f, indent=2)
            count += 1

    print(f"Successfully generated {count} Hamiltonian benchmark jobs in 'jobs/'.")
    print(f"Results will be written under: {args.results_dir}")


if __name__ == "__main__":
    main()
