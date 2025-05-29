import wandb
import pandas as pd
import numpy as np
import inspect
from collections import defaultdict
import os
import json

wandb_entity = "jaesikyoon"
wandb_project = "jaesik_mctd"

group_names = [
    "CS-PMCTD-Replanning",
    "CD-PMCTD-Replanning",
    "CT-PMCTD-Replanning",
    "CS-FMCTD-Replanning",
    "CD-FMCTD-Replanning",
    "CT-FMCTD-Replanning",
]

folder_name = "exp_results"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

api = wandb.Api()
group_data = {}
for group_name in group_names:
    if os.path.exists(f"{folder_name}/{group_name}.json"):
        with open(f"{folder_name}/{group_name}.json", "r") as f:
            group_data[group_name] = json.load(f)
        continue
    runs = api.runs(f"{wandb_entity}/{wandb_project}", filters={"group": group_name})
    _group_data = {}
    for run in runs:
        config = run.config
        summary = run.summary._json_dict
        _group_data[run.id] = {
            "config": run.config,
            "summary": run.summary._json_dict
        }
    with open(f"{folder_name}/{group_name}.json", "w") as f:
        json.dump(_group_data, f)
    group_data[group_name] = _group_data

# group, seed, task, success_rate, planning_time
dataframe = pd.DataFrame(columns=["group", "seed", "task", "success_rate", "planning_time"]) 
for group_name in group_names:
    for run_id, run_data in group_data[group_name].items():
        try:
            seed = run_data["config"]["experiment"]["validation"]["seed"]
            task = run_data["config"]["algorithm"]["task_id"]
            success_rate = run_data["summary"]["validation/success_rate"]
            planning_time = run_data["summary"]["validation/planning_time"]
            dataframe = pd.concat([dataframe, pd.DataFrame({
                "group": [group_name],
                "seed": [seed],
                "task": [task],
                "success_rate": [success_rate],
                "planning_time": [planning_time],
            })], ignore_index=True)
        except Exception as e:
            print(f"Error in {group_name} {run_id}: {e}")
            exit(1)

# group, success_rate, planning_time
summary_dataframe = pd.DataFrame(columns=["group", "success_rate", "planning_time"]) 
groups = dataframe["group"].unique()

for group in groups:
    # pick the best success rate for each seed and task_id
    success_rates = dataframe[dataframe["group"] == group].groupby(["seed", "task"])["success_rate"].max().mean()
    avg_sr, std_sr = success_rates.mean(), success_rates.std()
    avg_sr *= 100
    std_sr *= 100
    # the planning time is the mean of the picked samples
    idx_max = dataframe[dataframe["group"] == group].groupby(["seed", "task"])["success_rate"].idxmax()
    best_planning_times = dataframe.loc[idx_max, ["planning_time", "seed"]]
    planning_times = best_planning_times.groupby("seed")["planning_time"].mean()
    avg_pt, std_pt = planning_times.mean(), planning_times.std()
    summary_dataframe = pd.concat([summary_dataframe, pd.DataFrame({
        "group": [group],
        "success_rate": [f"{int(avg_sr)}±{int(std_sr)}"],
        "planning_time": [f"{avg_pt:.2f}±{std_pt:.2f}"],
    })], ignore_index=True)

for _data in summary_dataframe.to_dict(orient="records"):
    print(_data)
