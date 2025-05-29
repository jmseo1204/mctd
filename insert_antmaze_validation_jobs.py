import os
import copy
import json
from datetime import datetime

seeds = list(range(10))
task_ids = [1,2,3,4,5]

wandb_entity = "jaesikyoon"
wandb_project = "jaesik_mctd"

repeat_num = 3 # Three times running to get more robust results less affected by the DQL variance

basic_configs = [
    ##############################################################
    # Parallel MCTD
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "AMMN-PMCTD",
        "+name": "AMMN-PMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.open_loop_horizon": 1000,
        "algorithm.val_max_steps": 1000,
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "algorithm.warp_threshold": 0.5,
        "dataset": "og_antmaze_medium_navigate",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "8b3xf51l",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "AMLN-PMCTD",
        "+name": "AMLN-PMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.open_loop_horizon": 1000,
        "algorithm.val_max_steps": 1000,
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "dataset": "og_antmaze_large_navigate",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "4tapu6is",
    },
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
        "dataset": "og_antmaze_giant_navigate",
        "dataset.episode_len": 1000,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "pzt9dsm4",
    },
    ##############################################################
    # Fast MCTD
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "AMMN-FMCTD",
        "+name": "AMMN-FMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.guidance_scale": 3.0,
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 1000,
        "algorithm.val_max_steps": 1000,
        "algorithm.warp_threshold": 2.0,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "dataset": "og_antmaze_medium_navigate",
        "dataset.jump": 5,
        "dataset.episode_len": 100,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "eqqsopw2",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "AMLN-FMCTD",
        "+name": "AMLN-FMCTD",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.guidance_scale": 2.0,
        "algorithm.mctd": True,
        "algorithm.parallel_search_num": 200,
        "algorithm.open_loop_horizon": 1000,
        "algorithm.val_max_steps": 1000,
        "algorithm.warp_threshold": 2.0,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "dataset": "og_antmaze_large_navigate",
        "dataset.jump": 5,
        "dataset.episode_len": 100,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "5g4vp0wm",
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
        "load": "uzrq13fa",
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
                for _ in range(repeat_num):
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