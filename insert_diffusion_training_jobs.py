import os
import copy
import json
from datetime import datetime

wandb_entity = "jaesikyoon"
wandb_project = "jaesik_mctd"

configs = [
    #################################
    # PointMaze
    #################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "PointMazeMediumNavigate",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_maze2d_medium_navigate",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "PointMazeMediumNavigate-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_maze2d_medium_navigate",
        "dataset.jump": 5,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "PointMazeLargeNavigate",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_maze2d_large_navigate",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "PointMazeLargeNavigate-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_maze2d_large_navigate",
        "dataset.jump": 5,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "PointMazeGiantNavigate",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_maze2d_giant_navigate",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "PointMazeGiantNavigate-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_maze2d_giant_navigate",
        "dataset.jump": 5,
    },

    #################################
    # AntMaze
    #################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "AntMazeMediumNavigate",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_antmaze_medium_navigate",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "AntMazeMediumNavigate-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_antmaze_medium_navigate",
        "dataset.jump": 5,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "AntMazeLargeNavigate",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_antmaze_large_navigate",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "AntMazeLargeNavigate-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_antmaze_large_navigate",
        "dataset.jump": 5,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "AntMazeGiantNavigate",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_antmaze_giant_navigate",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "AntMazeGiantNavigate-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_antmaze_giant_navigate",
        "dataset.jump": 5,
    },
]

# Check there is the jobs folder
jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

# Write the jobs with each config, which name is current time (Too quickly to be overwritten)
for config in configs:
    with open(f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json", "w") as f:
        json.dump(config, f, indent=4)
print(f"Generated validation {len(configs)} jobs")