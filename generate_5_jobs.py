import os
import json
from datetime import datetime
import copy

# Use the AMGN-PMCTD config (giant maze) for testing
config_template = {
    "wandb.entity": "jmseo1204-seoul-national-university",
    "wandb.project": "mctd_eval",
    "wandb.group": "AMGN-PMCTD",
    "experiment": "exp_planning",
    "algorithm": "df_planning",
    "algorithm.open_loop_horizon": 1500,
    "algorithm.val_max_steps": 1500,
    "algorithm.parallel_search_num": 200,
    "algorithm.mctd_max_search_num": 500,
    "algorithm.mctd_guidance_scales": "[0,1,2,3,4,5]",
    "algorithm.sub_goal_interval": 10,
    "dataset": "og_antmaze_giant_navigate",
    "dataset.episode_len": 1000,
    "experiment.tasks": ["validation"],
    "experiment.validation.batch_size": 1,
    "load": "pzt9dsm4",
}

# Generate 5 jobs with different seeds and task_ids
jobs = []
for i, (seed, task_id) in enumerate([(1, 2), (2, 3), (3, 1), (4, 4), (5, 5)]):
    config = copy.deepcopy(config_template)
    config["+name"] = f"AMGN-PMCTD_TaskID{task_id}_Seed{seed}"
    config["experiment.validation.seed"] = seed
    config["algorithm.task_id"] = task_id
    
    jobs_folder = "jobs"
    if not os.path.exists(jobs_folder):
        os.makedirs(jobs_folder)
    
    with open(f"{jobs_folder}/job_{i:02d}.json", "w") as f:
        json.dump(config, f, indent=4)
    jobs.append(f"job_{i:02d}.json")

print(f"Generated {len(jobs)} jobs:")
for j in jobs:
    print(f"  - {j}")
