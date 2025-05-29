import os
import copy
import json
from datetime import datetime

wandb_entity = "jaesikyoon"
wandb_project = "jaesik_mctd"

configs = [
    #################################
    # Cube
    #################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "CubeSinglePlay",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_cube_single_play",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "CubeSinglePlay-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_cube_single_play",
        "dataset.jump": 5,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "CubeDoublePlay",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_cube_double_play",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "CubeDoublePlay-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_cube_double_play",
        "dataset.jump": 5,
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "CubeTriplePlay",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_cube_triple_play",
    },
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Training",
        "+name": "CubeTriplePlay-Interval5",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "dataset": "og_cube_triple_play",
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