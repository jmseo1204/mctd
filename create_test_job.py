import json
import os
from datetime import datetime

# Define the experiment configuration
experimental_config = {
    # Custom wandb settings
    "wandb.entity": "jmseo1204-seoul-national-university",
    "wandb.project": "mctd_eval",
    "wandb.group": "TEST-WANDB-CONFIG",
    "+name": "TEST_wandb_validation",

    # Experiment settings
    "experiment": "exp_planning",
    "algorithm": "df_planning",
    "algorithm.open_loop_horizon": 100,
    "algorithm.val_max_steps": 100,

    # Algorithm specific
    "algorithm.mctd": True,
    "algorithm.parallel_search_num": 10,
    "algorithm.mctd_max_search_num": 20,
    "algorithm.mctd_guidance_scales": "[0,1]",
    "algorithm.sub_goal_interval": 10,
    "algorithm.warp_threshold": 0.5,

    # Dataset and tasks
    "dataset": "og_antmaze_giant_navigate",
    "dataset.episode_len": 100,
    "experiment.tasks": ["validation"],
    
    # Run config
    "experiment.validation.batch_size": 1,
    "experiment.validation.seed": 0,
    "algorithm.task_id": 0,
    
    # NOTE: 'load' parameter removed! This is the fix.
}

jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

# Create a unique filename timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
job_filename = f"TEST_{timestamp}.json"
job_path = os.path.join(jobs_folder, job_filename)

with open(job_path, "w") as f:
    json.dump(experimental_config, f, indent=4)

print(f"Created test job: {job_path}")
print("Run 'python run_jobs.py' to execute this test job")
