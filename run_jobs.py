import subprocess
import time
import json
import os

available_gpus = ["localhost:0"]
# each server available gpus
# available_gpus += [f"rumelhart:{i}" for i in [0,1,2,3,4,5,6,7]]
# available_gpus += [f"levine:{i}" for i in [0,1,2,3,4,5,6,7]]

jobs_folder = "jobs"
docker_image = "fmctd:0.1"
docker_user = "jsyoon"
home_dir = os.path.expanduser("~")
project_dir = os.getcwd()
output_mount_dir = "/home/jmseo1204/mctd_outputs"
os.makedirs(output_mount_dir, exist_ok=True)
os.system(f"chmod 777 {output_mount_dir}")

# Dictionary to keep track of running experiments.
running_experiments = {gpu: None for gpu in available_gpus}

def start_experiment(server, gpu_id, config, exp_name, current_time):
    command_args = ""
    for key, value in config.items():
        command_args += f"{key}={value} "

    if server == "localhost":
        command = f"""
        docker run --rm -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
        -e MUJOCO_GL=osmesa \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {output_mount_dir}:/home/{docker_user}/mctd/outputs \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {home_dir}/.ogbench:/home/{docker_user}/.ogbench \
        {docker_image} /bin/bash \
        -c "cd mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}" \
        """
    else:
        if docker_user == "root":
            command = f"""
            ssh {server} docker run --rm -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
            -e MUJOCO_GL=osmesa \
            -v {project_dir}:/{docker_user}/mctd \
            -v {output_mount_dir}:/{docker_user}/mctd/outputs \
            -v {home_dir}/.netrc:/{docker_user}/.netrc \
            -v {home_dir}/.d4rl:/{docker_user}/.d4rl \
            -v {home_dir}/.ogbench:/{docker_user}/.ogbench \
            {docker_image} /bin/bash \
            -c "cd mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}" \
            """
        else:
            command = f"""
            ssh {server} docker run --rm -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
            -e MUJOCO_GL=osmesa \
            -v {project_dir}:/home/{docker_user}/mctd \
            -v {output_mount_dir}:/home/{docker_user}/mctd/outputs \
            -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
            -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
            -v {home_dir}/.ogbench:/home/{docker_user}/.ogbench \
            {docker_image} /bin/bash \
            -c "cd mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}" \
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

from tqdm import tqdm

# Need jobs folder exists
assert os.path.exists(jobs_folder), f"jobs folder does not exist"

# Get initial total number of jobs
total_jobs = len([f for f in os.listdir(jobs_folder) if f.endswith('.json')])
pbar = tqdm(total=total_jobs, desc="Processing Jobs")

# Check the jobs folder is empty or not
queue_is_empty = False
config_files = sorted(os.listdir(f"{jobs_folder}/"))
if config_files:
    config_file = config_files[0]
    # Read config file
    with open(f"{jobs_folder}/{config_file}", "r") as f:
        config = json.load(f)
else:
    queue_is_empty = True

completed_jobs = 0

try:
    while not queue_is_empty:
        for gpu, exp_name in list(running_experiments.items()):
            server, gpu_id = gpu.split(":")
            if exp_name is not None and not is_experiment_running(server, exp_name):
                running_experiments[gpu] = None  # Mark the GPU as available.
                completed_jobs += 1
                pbar.update(1)
                pbar.set_postfix({"Last Completed": exp_name})

            if running_experiments[gpu] is None:
                memory_used, memory_total = check_gpu_memory_usage(server, gpu_id)
                if memory_used < 1000: # If the memory usage is less than 500MB, start a new experiment.
                    current_time = time.strftime("%Y%m%d-%H%M%S")
                    exp_name = f"exp_gpu{gpu_id}_{current_time}-{jobs_folder}"  # Unique container name.
                    
                    # Check again if file exists (to avoid collision with another process)
                    if not os.path.exists(f"{jobs_folder}/{config_file}"):
                         config_files = sorted(os.listdir(f"{jobs_folder}/"))
                         if not config_files:
                             queue_is_empty = True
                             break
                         config_file = config_files[0]
                         with open(f"{jobs_folder}/{config_file}", "r") as f:
                             config = json.load(f)

                    start_experiment(server, gpu_id, config, exp_name, current_time)
                    running_experiments[gpu] = exp_name  # Set the container name.
                    
                    try:
                        os.remove(f"{jobs_folder}/{config_file}") # Remove the config file from the queue after running the experiment.
                    except FileNotFoundError:
                        pass

                    time.sleep(1) # Interval between starting containers.
                    config_files = sorted(os.listdir(f"{jobs_folder}/"))
                    if config_files:
                        config_file = config_files[0]
                        # Read config file
                        with open(f"{jobs_folder}/{config_file}", "r") as f:
                            config = json.load(f)
                    else:
                        queue_is_empty = True
                        break
        time.sleep(1)  # Check every 1 seconds.
except KeyboardInterrupt:
    print("\n\n!! KeyboardInterrupt detected. Cleaning up running docker containers... !!")
    for gpu, exp_name in running_experiments.items():
        if exp_name is not None:
            server, _ = gpu.split(":")
            print(f"Stopping container {exp_name} on {server}...")
            if server == "localhost":
                subprocess.run(["docker", "rm", "-f", exp_name], capture_output=True)
            else:
                subprocess.run(["ssh", server, "docker", "rm", "-f", exp_name], capture_output=True)
    print("Cleanup complete. Exiting.")

pbar.close()
print(f"\nAll {total_jobs} jobs finished!")