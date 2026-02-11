#!/usr/bin/env python3
import yaml
import subprocess
import os
import time

LOG_DIR = "experiment_logs"
CONFIG_PATH = "configurations/algorithm/df_planning.yaml"

def get_scales():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    scales = config.get('mctd_guidance_scales', [])
    return scales

def format_scales(scales):
    # Convert [0.0, 2.0, 8.0, 20.0, 40.0] -> "0_2_8_20_40"
    # Handling potential floats but keeping it clean
    parts = []
    for s in scales:
        if float(s).is_integer():
            parts.append(str(int(s)))
        else:
            parts.append(str(s).replace('.', '_'))
    return "_".join(parts)

def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    scales = get_scales()
    scales_str = format_scales(scales)
    log_name = f"scales_{scales_str}.log"
    log_path = os.path.join(LOG_DIR, log_name)

    # 1. Setup jobs
    print(f"Preparing experiment for scales: {scales}")
    # Clear existing jobs to ensure a clean run
    os.system("rm -rf jobs/*")
    # Insert new jobs
    os.system("python3 insert_giant_maze_validation_jobs.py")
    
    # Just print the path so the shell script can capture it
    print(f"LOG_FILE_PATH:{log_path}")

if __name__ == "__main__":
    main()
