import subprocess
import time
import json
import os
from tqdm import tqdm

available_gpus = ["localhost:0"]
# available_gpus += [f"levine:{i}" for i in [0,1,2,3,4,5,6,7]]

jobs_folder = "dql_jobs"
docker_image = "fmctd:0.1"
docker_user = "jsyoon"
home_dir = os.path.expanduser("~")
project_dir = os.getcwd()


# Dictionary to keep track of running experiments.
running_experiments = {gpu: None for gpu in available_gpus}

def start_experiment(server, gpu_id, config, exp_name, current_time):
    command_args = ""
    for key, value in config.items():
        command_args += f"--{key} {value} "

    if server == "localhost":
        command = f"""
        docker run --rm -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {home_dir}/.ogbench:/home/{docker_user}/.ogbench \
        {docker_image} /bin/bash \
        -c "cd mctd && python3 dql/main_Antmaze.py {command_args}" \
        """
    else:
        if docker_user == "root":
            command = f"""
            ssh {server} docker run --rm -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
            -v {project_dir}:/{docker_user}/mctd \
            -v {home_dir}/.netrc:/{docker_user}/.netrc \
            -v {home_dir}/.d4rl:/{docker_user}/.d4rl \
            -v {home_dir}/.ogbench:/{docker_user}/.ogbench \
            {docker_image} /bin/bash \
            -c "cd mctd && python3 dql/main_Antmaze.py {command_args}" \
            """
        else:
            command = f"""
            ssh {server} docker run --rm -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
            -v {project_dir}:/home/{docker_user}/mctd \
            -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
            -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
            -v {home_dir}/.ogbench:/home/{docker_user}/.ogbench \
            {docker_image} /bin/bash \
            -c "cd mctd && python3 dql/main_Antmaze.py {command_args}" \
            """
    os.system(command)

# Check GPU memory usage.
def check_gpu_memory_usage(server, gpu_id):
    # Execute nvidia-smi command to check GPU memory usage.
    if server == "localhost":
        command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}"
    else:
        command = f"ssh {server} nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    memory_used = result.stdout.strip()
    if memory_used == "":
        memory_used = 0
    memory_used = int(memory_used)
    # Execute nvidia-smi command to check GPU max memory.
    if server == "localhost":
        command = f"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_id}"
    else:
        command = f"ssh {server} nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_id}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    memory_total = result.stdout.strip()
    memory_total = int(memory_total)
    return memory_used, memory_total

# Function to check if a docker container is still running.
def is_experiment_running(server, exp_name):
    if server == "localhost":
        cmd = ["docker", "ps", "-q", "-f", f"name={exp_name}"]
    else:
        cmd = ["ssh", server, "docker", "ps", "-q", "-f", f"name={exp_name}"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    # If the result is not empty, the container is running.
    return bool(result.stdout.strip())

import re

def get_current_epoch(server, exp_name):
    if server == "localhost":
        cmd = f"docker logs --tail 100 {exp_name}"
    else:
        cmd = f"ssh {server} docker logs --tail 100 {exp_name}"
    
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # More robust regex to handle prefixes (e.g., "[antmaze-large...] Trained Epochs 16")
        matches = re.findall(r"Trained Epochs\s+(\d+)", result.stdout)
        if matches:
            return int(matches[-1])
    except:
        pass
    return None

# Get initial total number of jobs
config_files_list = [f for f in os.listdir(jobs_folder) if f.endswith('.json')]
total_jobs = len(config_files_list)
epochs_per_job = 200

# Main progress bar for total jobs
main_pbar = tqdm(total=total_jobs, desc="Total Jobs Progress", position=0)

# Dictionary to keep track of epoch progress bars for each GPU
job_pbars = {}
running_job_configs = {gpu: None for gpu in available_gpus}

# Check the jobs folder is empty or not
queue_is_empty = False
config_files = sorted(os.listdir(f"{jobs_folder}/"))
if config_files:
    config_file = config_files[0]
    with open(f"{jobs_folder}/{config_file}", "r") as f:
        config = json.load(f)
else:
    queue_is_empty = True

def find_existing_containers(gpu_id):
    """Find a running dql container on the given GPU id."""
    cmd = f"docker ps --format '{{{{.Names}}}}' --filter \"name=dql_\""
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    names = result.stdout.strip().split('\n')
    for name in names:
        if not name: continue
        # Verify if it's on the right GPU
        inspect_cmd = f"docker inspect {name} --format '{{{{.HostConfig.DeviceRequests}}}}'"
        inspect_res = subprocess.run(inspect_cmd, shell=True, stdout=subprocess.PIPE, text=True)
        if f"device={gpu_id}" in inspect_res.stdout:
            return name
    return None

while not queue_is_empty or any(running_experiments.values()):
    for i, (gpu, exp_name) in enumerate(list(running_experiments.items())):
        server, gpu_id = gpu.split(":")
        
        # 1. Try to auto-attach if we don't think something is running but a container exists
        if exp_name is None:
            existing = find_existing_containers(gpu_id)
            if existing:
                print(f"\nAuto-attached to existing container: {existing} on GPU {gpu_id}")
                running_experiments[gpu] = existing
                # Create a dummy config if we don't have it to show something in bar
                env_type = existing.split("_")[1] # medium/large/giant
                running_job_configs[gpu] = {"env_name": f"antmaze-{env_type}-navigate-v0"}
                exp_name = existing

        if exp_name is not None:
            if not is_experiment_running(server, exp_name):
                # Job just finished
                if gpu in job_pbars:
                    job_pbars[gpu].n = epochs_per_job
                    job_pbars[gpu].refresh()
                    job_pbars[gpu].close()
                    del job_pbars[gpu]
                
                running_experiments[gpu] = None
                running_job_configs[gpu] = None
                main_pbar.update(1)
            else:
                # Still running, try to get epoch
                epoch = get_current_epoch(server, exp_name)
                if epoch is not None:
                    if gpu not in job_pbars:
                        env_name = running_job_configs[gpu].get("env_name", "unknown") if running_job_configs[gpu] else "unknown"
                        job_pbars[gpu] = tqdm(total=epochs_per_job, desc=f" > GPU {gpu_id} ({env_name})", position=i+1, leave=False)
                    
                    job_pbars[gpu].n = epoch
                    job_pbars[gpu].refresh()

        if running_experiments[gpu] is None and not queue_is_empty:
            memory_used, memory_total = check_gpu_memory_usage(server, gpu_id)
            if memory_used < 1000:
                current_time = time.strftime("%Y%m%d-%H%M%S")
                env_tag = config.get("env_name", "").split("-")[1]
                new_exp_name = f"dql_{env_tag}_{current_time}"
                
                if not os.path.exists(f"{jobs_folder}/{config_file}"):
                     config_files = sorted(os.listdir(f"{jobs_folder}/"))
                     if not config_files:
                         queue_is_empty = True
                         break
                     config_file = config_files[0]
                     with open(f"{jobs_folder}/{config_file}", "r") as f:
                         config = json.load(f)

                start_experiment(server, gpu_id, config, new_exp_name, current_time)
                running_experiments[gpu] = new_exp_name
                running_job_configs[gpu] = config
                
                try:
                    os.remove(f"{jobs_folder}/{config_file}")
                except FileNotFoundError:
                    pass

                time.sleep(3)
                config_files = sorted(os.listdir(f"{jobs_folder}/"))
                if config_files:
                    config_file = config_files[0]
                    with open(f"{jobs_folder}/{config_file}", "r") as f:
                        config = json.load(f)
                else:
                    queue_is_empty = True

    time.sleep(10) # Log check interval

main_pbar.close()
print(f"\nAll {total_jobs} DQL jobs finished!")