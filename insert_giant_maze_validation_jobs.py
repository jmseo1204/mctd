import os
import copy
import json
from datetime import datetime

# seeds 0 to 9 to match antmaze script
seeds = list(range(3))
# Task IDs 1 to 5 for Giant Maze (Environment requirement)
task_ids = [1, 2, 3, 4, 5]

wandb_entity = "jmseo1204-seoul-national-university"
wandb_project = "mctd_eval"

# repeat_num 3 to match antmaze script for robustness
repeat_num = 1

# Proper IDs for Giant Maze from official configs
pmctd_giant_ckpt = "pzt9dsm4"
fmctd_giant_ckpt = "uzrq13fa"

basic_configs = [
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "AMGN-PMCTD",
        "+name": "AMGN-PMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.open_loop_horizon": 1500,
        "algorithm.val_max_steps": 1500,
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "algorithm.warp_threshold": 0.5,
        "dataset": "og_antmaze_giant_navigate",
        "dataset.episode_len": 1000,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": pmctd_giant_ckpt,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "AMGN-FMCTD",
        "+name": "AMGN-FMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.guidance_scale": 2.0,
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 1500,
        "algorithm.val_max_steps": 1500,
        "algorithm.warp_threshold": 2.0,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "dataset": "og_antmaze_giant_navigate",
        "dataset.jump": 5,
        "dataset.episode_len": 200,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": fmctd_giant_ckpt,
    },
]

configs = []
for config in basic_configs:
    name = copy.deepcopy(config["+name"])
    for task_id in task_ids:
        for seed in seeds:
            config_copy = copy.deepcopy(config)
            config_copy["experiment.validation.seed"] = seed
            config_copy["algorithm.task_id"] = task_id
            config_copy["+name"] = f"{name}_TaskID{task_id}_Seed{seed}"
            for _ in range(repeat_num):
                configs.append(config_copy)

jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

for config in configs:
    filename = f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json"
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)

print(f"Generated {len(configs)} Giant Maze jobs using correct Planner IDs and Task IDs [1-5]")
