import os
import copy
import json
from datetime import datetime

seeds = list(range(10))
task_ids = [1,2,3,4,5]

wandb_entity = "jaesikyoon"
wandb_project = "jaesik_mctd"

basic_configs = [
    ##############################################################
    # Parallel MCTD
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PMMN-PMCTD",
        "+name": "PMMN-PMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 500,
        "algorithm.val_max_steps": 500,
        "algorithm.mctd_max_search_num": 500,
        "dataset": "og_maze2d_medium_navigate",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "ynn5o8cb",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PMLN-PMCTD",
        "+name": "PMLN-PMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 500,
        "algorithm.val_max_steps": 500,
        "algorithm.mctd_max_search_num": 500,
        "dataset": "og_maze2d_large_navigate",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "5wy35u14",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PMGN-PMCTD",
        "+name": "PMGN-PMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 1000,
        "algorithm.val_max_steps": 1000,
        "algorithm.mctd_max_search_num": 500,
        "dataset": "og_maze2d_giant_navigate",
        "dataset.episode_len": 1000,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "71vbasu3",
    },
    ##############################################################
    # Fast MCTD
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PMMN-FMCTD",
        "+name": "PMMN-FMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 500,
        "algorithm.val_max_steps": 500,
        "algorithm.warp_threshold": 2.0,
        "algorithm.mctd_max_search_num": 500,
        "dataset": "og_maze2d_medium_navigate",
        "dataset.jump": 5,
        "dataset.episode_len": 100,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "veii4g8t",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PMLN-FMCTD",
        "+name": "PMLN-FMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 500,
        "algorithm.val_max_steps": 500,
        "algorithm.warp_threshold": 2.0,
        "algorithm.mctd_max_search_num": 500,
        "dataset": "og_maze2d_large_navigate",
        "dataset.jump": 5,
        "dataset.episode_len": 100,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "t2tlk0ca",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PMGN-FMCTD",
        "+name": "PMGN-FMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 1000,
        "algorithm.val_max_steps": 1000,
        "algorithm.warp_threshold": 2.0,
        "algorithm.mctd_max_search_num": 500,
        "dataset": "og_maze2d_giant_navigate",
        "dataset.jump": 5,
        "dataset.episode_len": 200,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "q940a89g",
    },
]

configs = []
for config in basic_configs:
    #group = copy.deepcopy(config["wandb.group"])
    name = copy.deepcopy(config["+name"])
    for task_id in task_ids:
            for seed in seeds:
                config = copy.deepcopy(config)
                config["experiment.validation.seed"] = seed
                config["algorithm.task_id"] = task_id
                config["+name"] = f"{name}_TaskID{task_id}_Seed{seed}"
                configs.append(config)

# Check there is the jobs folder
jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

# Write the jobs with each config, which name is current time (Too quickly to be overwritten)
for config in configs:
    with open(f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json", "w") as f:
        json.dump(config, f, indent=4)

print(f"Generated validation {len(configs)} jobs")