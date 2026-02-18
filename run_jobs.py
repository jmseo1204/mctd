import subprocess
import time
import json
import os
import shlex
import datetime
import yaml
from tqdm import tqdm

# Logging setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"run_{current_time_str}.log")
log_file = open(log_file_path, "w", buffering=1)

def log_write(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    # Mirror to log file
    log_file.write(f"{timestamp} {message}\n")
    # Output to pbar/terminal
    if 'pbar' in globals() and pbar is not None:
        pbar.write(message)
    else:
        print(message)

available_gpus = ["localhost:0"]
# each server available gpus
# available_gpus += [f"rumelhart:{i}" for i in [0,1,2,3,4,5,6,7]]
# available_gpus += [f"levine:{i}" for i in [0,1,2,3,4,5,6,7]]

jobs_folder = "jobs"
docker_image = "fmctd:0.1"
docker_user = "jsyoon"
home_dir = os.path.expanduser("~")
project_dir = os.getcwd()
ogbench_data_dir = os.path.abspath(os.path.join(project_dir, "..", "ogbench_data"))
output_mount_dir = "/home/jmseo1204/mctd_outputs"
os.makedirs(output_mount_dir, exist_ok=True)
os.system(f"chmod 777 {output_mount_dir}")

# Dictionary to keep track of running experiments.
running_experiments = {gpu: None for gpu in available_gpus}
last_log_line_count = {}

def get_og_dataset_name(dataset_config_name):
    """
    Look up the actual OGBench dataset filename from the YAML configuration.
    """
    yaml_path = os.path.join("configurations", "dataset", f"{dataset_config_name}.yaml")
    if not os.path.exists(yaml_path):
        return None
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # Recursively handle defaults if dataset key is missing in the leaf
            if 'dataset' in data:
                return data['dataset']
            # If not found, check base_dataset if applicable (though usually it's in the leaf)
            return None
    except Exception:
        return None

def start_experiment(server, gpu_id, config, exp_name, current_time, pbar):
    # Properly quote arguments for Hydra/Shell compatibility
    command_args = ""
    for key, value in config.items():
        # Handle lists/dicts as strings for the command line
        val_str = str(value).replace(" ", "")
        command_args += f"{shlex.quote(f'{key}={val_str}')} "

    if server == "localhost":
        command = f"""
        docker run -d --gpus '"device={gpu_id}"' --name {exp_name} --shm-size=50g \
        -e MUJOCO_GL=osmesa \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {output_mount_dir}:/home/{docker_user}/mctd/outputs \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {ogbench_data_dir}:/home/{docker_user}/.ogbench/data \
        {docker_image} /bin/bash \
        -c "git config --global --add safe.directory /home/{docker_user}/mctd && cd mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}"
        """
    else:
        # Multi-server setup example (ssh)
        command = f"""
        ssh {server} "docker run -d --gpus '\"device={gpu_id}\"' --name {exp_name} --shm-size=50g \
        -e MUJOCO_GL=osmesa \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {output_mount_dir}:/home/{docker_user}/mctd/outputs \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {ogbench_data_dir}:/home/{docker_user}/.ogbench/data \
        {docker_image} /bin/bash \
        -c 'git config --global --add safe.directory /home/{docker_user}/mctd && cd mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}'"
        """
        
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log_write(f"!! System Error starting {exp_name} on {server} !!")
        log_write(result.stderr.strip())
        return False
    
    # Background log harvester for this specific experiment
    if server == "localhost":
        # Start a background process to append this container's logs to our main log file
        log_cmd = f"docker logs -f {exp_name} >> {log_file_path} 2>&1 &"
        subprocess.Popen(log_cmd, shell=True)
    
    # Wait a moment and check if it's still alive
    time.sleep(0.5)
    if not is_experiment_running(server, exp_name):
        log_write(f"!! Container {exp_name} died immediately after start !!")
        try:
            # Try to get logs from the dead container
            if server == "localhost":
                logs = subprocess.run(["docker", "logs", exp_name], capture_output=True, text=True).stdout
            else:
                logs = subprocess.run(["ssh", server, "docker", "logs", exp_name], capture_output=True, text=True).stdout
            
            if logs:
                log_write("--- Container Logs ---")
                log_write(logs.strip())
        except:
            log_write("Could not retrieve logs from dead container.")
        return False
        
    return True

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
    else:
        try:
            memory_used = int(memory_used)
        except:
            memory_used = 0
    # Execute nvidia-smi command to check GPU max memory.
    if server == "localhost":
        command = f"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_id}"
    else:
        command = f"ssh {server} nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_id}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    memory_total = result.stdout.strip()
    try:
        memory_total = int(memory_total)
    except:
        memory_total = 8000 # dummy
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
    while not (queue_is_empty and all(v is None for v in running_experiments.values())):
        for gpu, exp_name in list(running_experiments.items()):
            server, gpu_id = gpu.split(":")
            
            # 1. Update status using docker logs
            if exp_name:
                try:
                    is_running = is_experiment_running(server, exp_name)
                    
                    if server == "localhost":
                        # Fetch logs (both stdout and stderr combined)
                        log_res = subprocess.run(["docker", "logs", "--tail", "15", exp_name], 
                                               capture_output=True, text=True)
                        # Combine stdout and stderr for full visibility
                        log_out = (log_res.stdout + log_res.stderr).strip()
                    else:
                        log_res = subprocess.run(["ssh", server, "docker", "logs", "--tail", "15", exp_name], 
                                               capture_output=True, text=True)
                        log_out = (log_res.stdout + log_res.stderr).strip()
                    
                    if log_out:
                        lines = log_out.split("\n")
                        last_seen = last_log_line_count.get(exp_name, "")
                        if lines[-1] != last_seen:
                            # If it's a new line, we might have multiple new lines
                            for line in lines[-5:]:
                                if line not in last_seen:
                                    log_write(f"[{gpu}] {line.strip()}")
                            last_log_line_count[exp_name] = lines[-1]
                    
                    # If the container died, make sure we see the final output
                    if not is_running and exp_name in running_experiments.values():
                        if log_out:
                            log_write(f"!! Container {exp_name} exited. Final logs: !!")
                            # Write last 5 lines of log_out
                            for line in log_out.split("\n")[-5:]:
                                log_write(f"[{gpu}] {line.strip()}")

                except Exception as e:
                    pass

            # 2. Check if finished
            if exp_name is not None and not is_experiment_running(server, exp_name):
                # Container just finished. Fetch final logs before removing it.
                try:
                    if server == "localhost":
                        log_res = subprocess.run(["docker", "logs", exp_name], capture_output=True, text=True)
                        final_logs = (log_res.stdout + log_res.stderr).strip()
                        subprocess.run(["docker", "rm", exp_name], capture_output=True)
                    else:
                        log_res = subprocess.run(["ssh", server, "docker", "logs", exp_name], capture_output=True, text=True)
                        final_logs = (log_res.stdout + log_res.stderr).strip()
                        subprocess.run(["ssh", server, "docker", "rm", exp_name], capture_output=True)
                    
                    # If it wasn't a clean exit or we want to see final status, print it
                    # Check if 'Finished' or similar success markers are in final_logs if needed
                    # For now, just print if there's an error/traceback
                    if "Traceback" in final_logs or "AssertionError" in final_logs or "Error" in final_logs:
                        log_write(f"!! Job {exp_name} failed. Final log snippet: !!")
                        for line in final_logs.split("\n")[-10:]:
                            log_write(f"[{gpu}] {line.strip()}")
                except:
                    pass

                running_experiments[gpu] = None  # Mark the GPU as available.
                completed_jobs += 1
                pbar.update(1)
                pbar.set_postfix({"Finished": exp_name})

            # 3. Start new if available
            if running_experiments[gpu] is None and not queue_is_empty:
                memory_used, memory_total = check_gpu_memory_usage(server, gpu_id)
                if memory_used < 1000: # If the memory usage is less than 1GB, start a new experiment.
                    current_time_job = time.strftime("%Y%m%d-%H%M%S")
                    exp_name = f"exp_gpu{gpu_id}_{current_time_job}-{jobs_folder}"
                    
                    if not os.path.exists(f"{jobs_folder}/{config_file}"):
                         config_files = sorted(os.listdir(f"{jobs_folder}/"))
                         if not config_files:
                             queue_is_empty = True
                             continue
                         config_file = config_files[0]
                         with open(f"{jobs_folder}/{config_file}", "r") as f:
                             config = json.load(f)

                    # --- Dataset Validation Check ---
                    dataset_config_name = config.get("dataset")
                    og_dataset_name = get_og_dataset_name(dataset_config_name)
                    
                    if og_dataset_name:
                        expected_npz = os.path.join(ogbench_data_dir, f"{og_dataset_name}.npz")
                        if not os.path.exists(expected_npz):
                            log_write(f"!! Error: Dataset file '{og_dataset_name}.npz' NOT FOUND in {ogbench_data_dir} !!")
                            log_write(f"!! Skipping job {config_file} to prevent automatic download !!")
                            try:
                                os.remove(f"{jobs_folder}/{config_file}")
                            except:
                                pass
                            
                            # Move to next job if available
                            config_files = sorted(os.listdir(f"{jobs_folder}/"))
                            if config_files:
                                config_file = config_files[0]
                                continue
                            else:
                                queue_is_empty = True
                                break

                    if start_experiment(server, gpu_id, config, exp_name, current_time_job, pbar):
                        running_experiments[gpu] = exp_name
                        last_log_line_count[exp_name] = 0
                        
                        try:
                            os.remove(f"{jobs_folder}/{config_file}")
                        except FileNotFoundError:
                            pass

                        time.sleep(1)
                    else:
                        # If failed to start, the error is already printed. 
                        # We don't remove the job, just try again next loop or wait.
                        pass
                    config_files = sorted(os.listdir(f"{jobs_folder}/"))
                    if config_files:
                        config_file = config_files[0]
                        with open(f"{jobs_folder}/{config_file}", "r") as f:
                            config = json.load(f)
                    else:
                        queue_is_empty = True
        time.sleep(2)
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