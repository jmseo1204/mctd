
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from project_config import WANDB_ENTITY, WANDB_PROJECT

# Define configurations for Maze2D and AntMaze Stitch (Medium) datasets
# Based on insert_pointmaze_validation_jobs.py and insert_antmaze_validation_jobs.py
# "Stitch" usually corresponds to Medium/Large datasets in offline RL benchmarks.

wandb_entity = WANDB_ENTITY
wandb_project = WANDB_PROJECT

configs = [
    # Maze2D (PointMaze) Medium - Often considered a stitch task in D4RL context (multimodal data)
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Reproduction-Maze2D-Medium",
        "+name": "Reproduction-Maze2D-Medium",
        "experiment": "base_pytorch",
        "algorithm": "df_planning",
        "algorithm.parallel_search_num": 50, # Reduced for quick check
        "algorithm.open_loop_horizon": 500,
        "algorithm.val_max_steps": 500,
        "algorithm.mctd_max_search_num": 100, # Reduced for quick check
        "dataset": "og_maze2d_medium_navigate",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "experiment.validation.precision": 32,
        "experiment.validation.inference_mode": False,
        "experiment.validation.limit_batch": 1,
        "load": "ynn5o8cb", # Run ID from the original file
    },
    # AntMaze Medium - The classic stitch dataset
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "Reproduction-AntMaze-Medium",
        "+name": "Reproduction-AntMaze-Medium",
        "experiment": "base_pytorch",
        "algorithm": "df_planning",
        "algorithm.open_loop_horizon": 500, # Reduced horizon for quick check? No, keep logic similar.
        "algorithm.val_max_steps": 1000,
        "algorithm.parallel_search_num": 50,
        "algorithm.mctd_max_search_num": 100,
        "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
        "algorithm.sub_goal_interval": 10,
        "algorithm.warp_threshold": 0.5,
        "dataset": "og_antmaze_medium_navigate",
        "dataset.episode_len": 500,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "experiment.validation.precision": 32,
        "experiment.validation.inference_mode": False,
        "experiment.validation.limit_batch": 1,
        "load": "8b3xf51l", # Run ID from the original file
    }
]

jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

generated_files = []
for i, config in enumerate(configs):
    filename = f"{jobs_folder}/reproduce_job_{i}.json"
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)
    generated_files.append(filename)
    print(f"Generated job file: {filename}")

print("\nTo run these jobs, you would typically use run_jobs.py.")
print("However, without the pre-trained models downloaded from W&B, `main.py` will attempt to download them from W&B.")
print("Since we likely lack W&B credentials for the original project, this might fail or require login.")

# Attempt to run one job to demonstrate the reproduction step
print("\nAttempting to run the Maze2D reproduction job (dry run)...")
cmd = ["python", "main.py", "name=test_repro", "wandb.mode=disabled"]
# We need to pass the config values as arguments to main.py (overriding defaults)
# But main.py reads from config.yaml. The job files are usually read by run_jobs.py which spawns processes.

# Construct arguments for main.py based on the first config
config = configs[0]
args = []
for k, v in config.items():
    if k.startswith("+"): # Hydra syntax for new keys
        args.append(f"{k}={v}")
    elif k in ["load", "wandb.entity", "wandb.project", "wandb.group"]: # Top level or specific overwrites
         args.append(f"{k}={v}")
    else:
        # Hydra nested keys
        args.append(f"{k}={v}")

# Default overrides to ensure it runs
args.append("wandb.mode=disabled") 

full_cmd = " ".join(cmd + args)
print(f"Executing: {full_cmd}")

# Note: We are not actually running it here because it will hang trying to download the model.
# I will create a shell script to run it instead.
