import json
import os
from datetime import datetime

# Define the experiment configuration - simpler version
experimental_config = {
    # Experiment settings
    "experiment": "exp_planning",
    "algorithm": "df_planning",
    
    # Algorithm specific
    "algorithm.open_loop_horizon": 50,
    "algorithm.parallel_search_num": 5,
    "algorithm.mctd_max_search_num": 10,
    "algorithm.mctd_guidance_scales": "[0,1]",
    "algorithm.sub_goal_interval": 10,

    # Dataset and tasks
    "dataset": "og_antmaze_giant_navigate",
    "experiment.tasks": ["validation"],
    
    # Run config
    "experiment.validation.batch_size": 1,
    "experiment.validation.seed": 0,
}

jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
job_filename = f"DIM_CHECK_{timestamp}.json"
job_path = os.path.join(jobs_folder, job_filename)

with open(job_path, "w") as f:
    json.dump(experimental_config, f, indent=4)

print(f"Created dimension check job: {job_path}")
