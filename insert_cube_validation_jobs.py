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
    # Parallel MCTD Replanning
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "CS-PMCTD-Replanning",
        "+name": "CS-PMCTD-Replanning",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.open_loop_horizon": 100,
        "algorithm.val_max_steps": 200,
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_guidance_scales": "[1,2,4]",
        "algorithm.mctd_max_search_num": 500,
        "algorithm.sub_goal_threshold": 0.15,
        "algorithm.warp_threshold": 1.0,
        "algorithm.cube_viz": True,
        "dataset": "og_cube_single_play",
        "dataset.episode_len": 200,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "x3rlj069",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "CD-PMCTD-Replanning",
        "+name": "CD-PMCTD-Replanning",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.open_loop_horizon": 200,
        "algorithm.val_max_steps": 500,
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_guidance_scales": "[1,2,4]",
        "algorithm.mctd_max_search_num": 500,
        "algorithm.sub_goal_threshold": 0.15,
        "algorithm.cube_viz": True,
        "algorithm.cube_single_dql": True,
        "dataset": "og_cube_double_play",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "xn2tbc3m",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "CT-PMCTD-Replanning",
        "+name": "CT-PMCTD-Replanning",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.open_loop_horizon": 200,
        "algorithm.val_max_steps": 500,
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.mctd_guidance_scales": "[1,2,4]",
        "algorithm.sub_goal_threshold": 0.15,
        "algorithm.cube_viz": True,
        "algorithm.cube_single_dql": True,
        "dataset": "og_cube_triple_play",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "wouak5pu",
    },

    ##############################################################
    # Fast MCTD
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "CS-FMCTD-Replanning",
        "+name": "CS-FMCTD-Replanning",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.open_loop_horizon": 100,
        "algorithm.val_max_steps": 200,
        "algorithm.mctd_guidance_scales": "[1,2,4]",
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.sub_goal_threshold": 0.15,
        "algorithm.warp_threshold": 2.0,
        "algorithm.cube_viz": True,
        "algorithm.cube_single_dql": True,
        "dataset": "og_cube_single_play",
        "dataset.jump": 5,
        "dataset.episode_len": 40,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "smks7g49",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "CD-FMCTD-Replanning",
        "+name": "CD-FMCTD-Replanning",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.open_loop_horizon": 200,
        "algorithm.val_max_steps": 500,
        "algorithm.mctd_guidance_scales": "[1,2,4]",
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.sub_goal_threshold": 0.15,
        "algorithm.warp_threshold": 2.0,
        "algorithm.cube_viz": True,
        "algorithm.cube_single_dql": True,
        "dataset": "og_cube_double_play",
        "dataset.jump": 5,
        "dataset.episode_len": 100,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "4nvelncd",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "CT-FMCTD-Replanning",
        "+name": "CT-FMCTD-Replanning",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": True,
        "algorithm.open_loop_horizon": 200,
        "algorithm.val_max_steps": 500,
        "algorithm.mctd_guidance_scales": "[1,2,4]",
        "algorithm.parallel_search_num": 200,
        "algorithm.mctd_max_search_num": 500,
        "algorithm.sub_goal_threshold": 0.15,
        "algorithm.warp_threshold": 2.0,
        "algorithm.cube_viz": True,
        "algorithm.cube_single_dql": True,
        "dataset": "og_cube_triple_play",
        "dataset.jump": 5,
        "dataset.episode_len": 100,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "xp9ts3mr",
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