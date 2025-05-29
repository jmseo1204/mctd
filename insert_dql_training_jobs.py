import os
import copy
import json
from datetime import datetime

configs = [
    {"env_name": "cube-single-play-v0"},
]

# Check there is the jobs folder
jobs_folder = "dql_jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

# Write the jobs with each config, which name is current time (Too quickly to be overwritten)
for config in configs:
    with open(f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json", "w") as f:
        json.dump(config, f, indent=4)
print(f"Generated {len(configs)} jobs")