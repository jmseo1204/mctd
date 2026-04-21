import argparse
import copy
import json
import os
from datetime import datetime

from project_config import WANDB_ENTITY, WANDB_PROJECT


def _parse_bool_arg(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


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
        "--repeat_ids",
        default=None,
        help="Optional comma-separated repeat ids to emit. Defaults to 1..num_repeats.",
    )
    parser.add_argument(
        "--results_file_prefix",
        default="hamiltonian_benchmark",
        help="Filename prefix for per-task benchmark JSON results",
    )
    parser.add_argument(
        "--planner_variant",
        default="online_hemiltonian",
        help="Human-readable planner variant name used in metrics/media/result payloads.",
    )
    parser.add_argument(
        "--route_mode",
        default="online",
        choices=("online", "fixed_temporal"),
        help="Hamiltonian route policy used during multi-tree planning.",
    )
    parser.add_argument(
        "--plan_only",
        default="false",
        help="If true, stop after plan generation/visualization and skip env rollout.",
    )
    parser.add_argument(
        "--log_groundtruth_media",
        default="true",
        help="If false, skip GT media logging while still computing GT stats.",
    )
    parser.add_argument(
        "--postprocessed_plan_media_key",
        default="postprocessed_plan",
        help="WandB media key used for the variant's postprocessed plan image.",
    )
    parser.add_argument(
        "--shared_run_id_base",
        default=None,
        help="If set, enable WandB shared mode and derive per-job shared ids from this base.",
    )
    parser.add_argument(
        "--shared_primary",
        default="true",
        help="Whether this job should act as the primary shared-mode process.",
    )
    parser.add_argument(
        "--shared_label",
        default=None,
        help="Optional shared-mode WandB label for logs/system metrics.",
    )
    parser.add_argument(
        "--batch_tasks_per_repeat",
        default="false",
        help="If true, emit one job per repeat with +algorithm.task_ids instead of one job per task.",
    )
    parser.add_argument(
        "--ogbench-enable-reset-perturb",
        default=None,
        help="Whether OGBench resets should keep start/goal perturbation during benchmark runtime (true/false).",
    )
    args = parser.parse_args()

    if args.waypoint_top_n <= 0:
        raise ValueError(f"--waypoint_top_n must be positive, got {args.waypoint_top_n}")
    ogbench_enable_reset_perturb = None
    if args.ogbench_enable_reset_perturb is not None:
        ogbench_enable_reset_perturb = _parse_bool_arg(args.ogbench_enable_reset_perturb)
    plan_only = _parse_bool_arg(args.plan_only)
    log_groundtruth_media = _parse_bool_arg(args.log_groundtruth_media)
    shared_primary = _parse_bool_arg(args.shared_primary)
    batch_tasks_per_repeat = _parse_bool_arg(args.batch_tasks_per_repeat)
    if args.repeat_ids:
        repeat_ids = []
        for raw_repeat_id in str(args.repeat_ids).split(","):
            raw_repeat_id = raw_repeat_id.strip()
            if not raw_repeat_id:
                continue
            repeat_id = int(raw_repeat_id)
            if repeat_id <= 0:
                raise ValueError(f"--repeat_ids entries must be positive, got {repeat_id}")
            repeat_ids.append(repeat_id)
        if not repeat_ids:
            raise ValueError("--repeat_ids was provided but no valid ids were parsed")
    else:
        repeat_ids = list(range(1, args.num_repeats + 1))

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
        "algorithm.multi_tree_route_mode": args.route_mode,
        "algorithm.task_override_path": args.task_override_path,
        "algorithm.benchmark_waypoint_top_n": int(args.waypoint_top_n),
        "algorithm.benchmark_model_id": args.model_id,
        "algorithm.benchmark_variant_name": args.planner_variant,
        "algorithm.benchmark_plan_only": bool(plan_only),
        "algorithm.benchmark_log_groundtruth_media": bool(log_groundtruth_media),
        "algorithm.benchmark_postprocessed_plan_media_key": args.postprocessed_plan_media_key,
        "wandb.job_type": args.planner_variant,
    }
    if args.planning_config_snapshot:
        basic_job_config["+algorithm_snapshot_path"] = args.planning_config_snapshot
    if ogbench_enable_reset_perturb is not None:
        basic_job_config["algorithm.ogbench_enable_reset_perturb"] = bool(ogbench_enable_reset_perturb)
    if args.shared_run_id_base:
        basic_job_config["wandb.shared_mode"] = True
        basic_job_config["wandb.shared_primary"] = bool(shared_primary)
        basic_job_config["wandb.shared_update_finish_state"] = bool(shared_primary)
        if args.shared_label:
            basic_job_config["wandb.shared_label"] = args.shared_label

    count = 0
    for repeat_id in repeat_ids:
        shared_run_id = None
        if args.shared_run_id_base:
            shared_run_id = f"{args.shared_run_id_base}-r{repeat_id:02d}"

        if batch_tasks_per_repeat:
            job_cfg = copy.deepcopy(basic_job_config)
            job_cfg["algorithm.eval_repeat_id"] = repeat_id
            job_cfg["+algorithm.task_ids"] = list(range(1, args.num_tasks + 1))
            job_cfg["algorithm.benchmark_rollout_seed_base"] = repeat_id * 100000 + 1000
            job_cfg["algorithm.benchmark_results_path"] = os.path.join(
                args.results_dir,
                f"{args.results_file_prefix}_repeat_{repeat_id:02d}.json",
            )
            job_cfg["+name"] = f"HBENCH_{args.planner_variant}_{args.model_id}_R{repeat_id}"
            if shared_run_id is not None:
                job_cfg["wandb.shared_id"] = shared_run_id

            filename = f"jobs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(job_cfg, f, indent=2)
            count += 1
            continue

        for task_id in range(1, args.num_tasks + 1):
            job_cfg = copy.deepcopy(basic_job_config)
            job_cfg["algorithm.task_id"] = task_id
            job_cfg["algorithm.eval_repeat_id"] = repeat_id
            job_cfg["algorithm.benchmark_rollout_seed_base"] = repeat_id * 100000 + task_id * 1000
            job_cfg["algorithm.benchmark_results_path"] = os.path.join(
                args.results_dir,
                f"{args.results_file_prefix}_repeat_{repeat_id:02d}_task_{task_id:02d}.json",
            )
            job_cfg["+name"] = f"HBENCH_{args.planner_variant}_{args.model_id}_R{repeat_id}_T{task_id}"
            if shared_run_id is not None:
                job_cfg["wandb.shared_id"] = f"{shared_run_id}-t{task_id:02d}"

            filename = f"jobs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(job_cfg, f, indent=2)
            count += 1

    print(f"Successfully generated {count} Hamiltonian benchmark jobs in 'jobs/'.")
    print(f"Results will be written under: {args.results_dir}")


if __name__ == "__main__":
    main()
