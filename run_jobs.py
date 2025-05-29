import subprocess
import time
import json
import os

available_gpus = []
# each server available gpus
available_gpus += [f"rumelhart:{i}" for i in [0,1,2,3,4,5,6,7]]
available_gpus += [f"levine:{i}" for i in [0,1,2,3,4,5,6,7]]
#available_gpus += [f"mcclelland:{i}" for i in [0,1,2,3,4,5,6,7]]

jobs_folder = "jobs"
docker_image = "jsyoon/mctd:0.1"
docker_user = "jsyoon"
home_dir = "/data/jsyoon"
project_dir = f"{home_dir}/projects/mctd"

# Dictionary to keep track of running experiments.
running_experiments = {gpu: None for gpu in available_gpus}

def start_experiment(server, gpu_id, config, exp_name, current_time):
    command_args = ""
    for key, value in config.items():
        command_args += f"{key}={value} "
    if docker_user == "root":
        command = f"""
        ssh {server} docker run -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
        -v {project_dir}:/{docker_user}/mctd \
        -v {home_dir}/.netrc:/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/{docker_user}/.d4rl \
        -v {home_dir}/.ogbench:/{docker_user}/.ogbench \
        {docker_image} /bin/bash \
        -c '"cd mctd; python3 main.py hostname={server} gpu_id={gpu_id} {command_args} "' \
        """
    else:
        command = f"""
        ssh {server} docker run -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {home_dir}/.ogbench:/home/{docker_user}/.ogbench \
        {docker_image} /bin/bash \
        -c '"cd mctd; python3 main.py hostname={server} gpu_id={gpu_id} {command_args} "' \
        """
    os.system(command)

# Check GPU memory usage.
def check_gpu_memory_usage(server, gpu_id):
    # Execute nvidia-smi command to check GPU memory usage.
    command = f"ssh {server} nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    memory_used = result.stdout.strip()
    if memory_used == "":
        memory_used = 0
    memory_used = int(memory_used)
    # Execute nvidia-smi command to check GPU max memory.
    command = f"ssh {server} nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_id}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    memory_total = result.stdout.strip()
    memory_total = int(memory_total)
    return memory_used, memory_total

# Function to check if a docker container is still running.
def is_experiment_running(server, exp_name):
    result = subprocess.run(
        ["ssh", server, "docker", "ps", "-q", "-f", f"name={exp_name}"],
        stdout=subprocess.PIPE
    )
    # If the result is not empty, the container is running.
    return bool(result.stdout.strip())

# Need jobs folder exists
assert os.path.exists(jobs_folder), f"jobs folder does not exist"

# Check the jobs folder is empty or not
queue_is_empty = False
config_files = sorted(os.listdir(f"{jobs_folder}/"))
if config_files:
    config_file = config_files[0]
    print(f"Starting experiment with config: {config_file}")
    # Read config file
    with open(f"{jobs_folder}/{config_file}", "r") as f:
        config = json.load(f)
else:
    queue_is_empty = True

while not queue_is_empty:
    for gpu, exp_name in list(running_experiments.items()):
        server, gpu_id = gpu.split(":")
        if exp_name is not None and not is_experiment_running(server, exp_name):
            running_experiments[gpu] = None  # Mark the GPU as available.
        if running_experiments[gpu] is None:
            memory_used, memory_total = check_gpu_memory_usage(server, gpu_id)
            if memory_used < 100: # If the memory usage is less than 500MB, start a new experiment.
            #if (memory_total - memory_used) > 1000: # If the remaining memory is more than 1000MB, start a new experiment.
                current_time = time.strftime("%Y%m%d-%H%M%S")
                exp_name = f"exp_gpu{gpu_id}_{current_time}-{jobs_folder}"  # Unique container name.
                print(f"Starting experiment on GPU {gpu} with config: {config}")
                start_experiment(server, gpu_id, config, exp_name, current_time)
                running_experiments[gpu] = exp_name  # Set the container name.
                os.remove(f"{jobs_folder}/{config_file}") # Remove the config file from the queue after running the experiment.
                time.sleep(1) # Interval between starting containers.
                config_files = sorted(os.listdir(f"{jobs_folder}/"))
                if config_files:
                    config_file = config_files[0]
                    print(f"Starting experiment with config: {config_file}")
                    # Read config file
                    with open(f"{jobs_folder}/{config_file}", "r") as f:
                        config = json.load(f)
                else:
                    queue_is_empty = True
                    break
    time.sleep(1)  # Check every 1 seconds.