import subprocess
import time
import json
import os
import sys
import shlex
import datetime
import threading
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


def log_finished(step_name):
    log_write(f"##### {step_name} finished! #####")

available_gpus = ["localhost:0"]
# each server available gpus
# available_gpus += [f"rumelhart:{i}" for i in [0,1,2,3,4,5,6,7]]
# available_gpus += [f"levine:{i}" for i in [0,1,2,3,4,5,6,7]]

jobs_folder = "jobs"
docker_image = "mctd:0.1"
docker_user = "jmseo1204"
home_dir = os.path.expanduser("~")
project_dir = os.getcwd()
ogbench_data_dir = os.path.abspath(os.path.join(project_dir, "..", "ogbench_data"))
hilp_dir = os.path.abspath(os.path.join(project_dir, "..", "HILP"))
jax_cache_dir = os.path.expanduser("~/.jax_cache")
os.makedirs(jax_cache_dir, exist_ok=True)
os.makedirs(os.path.join(jax_cache_dir, "xla_gpu_per_fusion_autotune_cache_dir"), exist_ok=True)
output_mount_dir = "/home/jmseo1204/mctd_outputs"
os.makedirs(output_mount_dir, exist_ok=True)
os.system(f"chmod 777 {output_mount_dir}")
# Ensure today's date dir is writable by Docker (uid 1020) if already created by host user
today_dir = os.path.join(output_mount_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
os.makedirs(today_dir, exist_ok=True)
os.system(f"chmod 777 {today_dir}")

# Dictionary to keep track of running experiments.
running_experiments = {gpu: None for gpu in available_gpus}
last_log_line_count = {}
last_log_time = {}       # exp_name -> time.time() of last log line seen
log_streamer_threads = {}  # exp_name -> Thread


def _stream_logs_worker(exp_name, server):
    """Stream container logs to terminal + log file in real-time."""
    cmd = (["docker", "logs", "-f", exp_name] if server == "localhost"
           else ["ssh", server, "docker", "logs", "-f", exp_name])
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            log_file.write(f"{ts} [{exp_name}] {line}\n")
            last_log_time[exp_name] = time.time()
            msg = f"[{exp_name[:30]}] {line}"
            if 'pbar' in globals() and pbar is not None:
                pbar.write(msg)
            else:
                print(msg, flush=True)
        proc.wait()
    except Exception:
        pass

def preprocess_batch_jobs(folder: str) -> None:
    """Group jobs with identical config (except task_id) into single batched jobs.

    Jobs that only differ in task_id are merged into one file with a task_ids list,
    so one Docker container handles all tasks and pays startup cost only once.
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(folder, "*.json")))
    if not files:
        return

    # Separate out files that already have task_ids (already batched)
    groups = {}  # canonical_key -> list of (fpath, task_id, base_config)
    already_batched = []
    for fpath in files:
        with open(fpath) as f:
            config = json.load(f)
        if "task_ids" in config:
            already_batched.append(fpath)
            continue
        task_id = config.pop("task_id", None)
        if task_id is None:
            continue
        key = json.dumps(config, sort_keys=True)
        groups.setdefault(key, []).append((fpath, task_id, config))

    merged_count = 0
    for key, items in groups.items():
        # Restore task_id for single-job groups
        if len(items) == 1:
            fpath, task_id, config = items[0]
            config["task_id"] = task_id
            with open(fpath, "w") as f:
                json.dump(config, f, indent=2)
            continue

        # Merge: first file becomes the batched job; others are deleted
        items_sorted = sorted(items, key=lambda x: x[1])  # sort by task_id
        fpath_keep, _, base_config = items_sorted[0]
        base_config["task_ids"] = [tid for _, tid, _ in items_sorted]
        with open(fpath_keep, "w") as f:
            json.dump(base_config, f, indent=2)
        for fpath, _, _ in items_sorted[1:]:
            os.remove(fpath)
        merged_count += 1

    if merged_count > 0:
        total_merged = sum(len(items) for items in groups.values() if len(items) > 1)
        print(f"[JobBatcher] Batched {total_merged} task jobs → {merged_count} merged jobs (startup cost paid once per batch)")


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
    # Keys not present in any yaml config must be prefixed with '+' for Hydra
    _EXTRA_KEYS = {"task_ids"}
    command_args = ""
    for key, value in config.items():
        # Handle lists/dicts as strings for the command line
        val_str = str(value).replace(" ", "")
        prefix = "+" if key in _EXTRA_KEYS else ""
        command_args += f"{shlex.quote(f'{prefix}{key}={val_str}')} "

    if server == "localhost":
        command = f"""
        docker run -d --gpus all --name {exp_name} --shm-size=50g \
        -e MUJOCO_GL=osmesa \
        -e HYDRA_FULL_ERROR=1 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e WANDB_EXIT_TIMEOUT=120 \
        -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/jmseo1204/.mujoco/mujoco210/bin \
        -v /usr/lib/wsl:/usr/lib/wsl \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {output_mount_dir}:/home/{docker_user}/mctd/outputs \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {ogbench_data_dir}:/home/{docker_user}/.ogbench/data \
        -v {hilp_dir}:/home/{docker_user}/HILP \
        -v {jax_cache_dir}:/home/{docker_user}/.jax_cache \
        {docker_image} /bin/bash \
        -c "git config --global --add safe.directory /home/{docker_user}/mctd && cd /home/{docker_user}/mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}"
        """
    else:
        # Multi-server setup example (ssh)
        command = f"""
        ssh {server} "docker run -d --gpus all --name {exp_name} --shm-size=50g \
        -e MUJOCO_GL=osmesa \
        -e HYDRA_FULL_ERROR=1 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -v {project_dir}:/home/{docker_user}/mctd \
        -v {output_mount_dir}:/home/{docker_user}/mctd/outputs \
        -v {home_dir}/.netrc:/home/{docker_user}/.netrc \
        -v {home_dir}/.d4rl:/home/{docker_user}/.d4rl \
        -v {ogbench_data_dir}:/home/{docker_user}/.ogbench/data \
        -v {hilp_dir}:/home/{docker_user}/HILP \
        -v {jax_cache_dir}:/home/{docker_user}/.jax_cache \
        {docker_image} /bin/bash \
        -c 'git config --global --add safe.directory /home/{docker_user}/mctd && cd /home/{docker_user}/mctd && python3 main.py hostname={server} gpu_id={gpu_id} {command_args}'"
        """
        
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log_write(f"!! System Error starting {exp_name} on {server} !!")
        log_write(result.stderr.strip())
        return False
    
    # Start real-time log streaming thread (shows in terminal + writes to log file)
    t = threading.Thread(target=_stream_logs_worker, args=(exp_name, server), daemon=True)
    t.start()
    log_streamer_threads[exp_name] = t
    last_log_time[exp_name] = time.time()
    
    # Wait a moment and check if it's still alive
    time.sleep(0.5)
    if not is_experiment_running(server, exp_name):
        log_write(f"!! Container {exp_name} died immediately after start !!")
        try:
            # Try to get logs from the dead container
            if server == "localhost":
                log_res = subprocess.run(["docker", "logs", exp_name], capture_output=True, text=True)
                logs = log_res.stdout + log_res.stderr
            else:
                log_res = subprocess.run(["ssh", server, "docker", "logs", exp_name], capture_output=True, text=True)
                logs = log_res.stdout + log_res.stderr
            
            if logs:
                log_write("--- Container Logs (stdout+stderr) ---")
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

# Merge jobs that only differ in task_id so one container handles all tasks
preprocess_batch_jobs(jobs_folder)
log_finished("job batching")

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
MAX_JOB_RETRIES = 3  # skip a job after this many consecutive immediate failures
job_fail_counts = {}  # config_file -> consecutive failure count

try:
    while not (queue_is_empty and all(v is None for v in running_experiments.values())):
        for gpu, exp_name in list(running_experiments.items()):
            server, gpu_id = gpu.split(":")
            
            # 1. Show elapsed time when no new log lines appear (silent wandb finalization)
            if exp_name:
                silence_sec = time.time() - last_log_time.get(exp_name, time.time())
                if silence_sec > 30 and int(silence_sec) % 30 == 0:
                    log_write(f"[{gpu}] Still running... ({silence_sec:.0f}s since last output — wandb finalizing)")

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
                    
                    # Detect real failures: Traceback present but exclude Hydra's normal
                    # "Error executing job with overrides" summary line (not a crash).
                    has_traceback = "Traceback (most recent call last)" in final_logs
                    if has_traceback:
                        log_write(f"!! Job {exp_name} failed. Final log snippet: !!")
                        for line in final_logs.split("\n")[-15:]:
                            log_write(f"[{gpu}] {line.strip()}")
                        # Exit immediately on error
                        log_write(f"[ERROR] Exiting due to job failure: {exp_name}")
                        sys.exit(1)
                except:
                    pass

                running_experiments[gpu] = None  # Mark the GPU as available.
                completed_jobs += 1
                pbar.update(1)
                pbar.set_postfix({"Finished": exp_name})
                log_finished(f"job {exp_name}")

            # 3. Start new if available
            if running_experiments[gpu] is None and not queue_is_empty:
                memory_used, memory_total = check_gpu_memory_usage(server, gpu_id)
                # Bypass memory check if retrying a recently failed job (crashed container
                # may leave GPU memory temporarily elevated even though GPU is free)
                current_fail_count = job_fail_counts.get(config_file, 0)
                memory_ok = memory_used < 2000 or current_fail_count > 0
                if memory_ok: # If GPU memory is free (or retrying a failed job), start a new experiment.
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
                        job_fail_counts.pop(config_file, None)  # reset on success
                        task_desc = config.get("task_ids", config.get("task_id", "unknown"))
                        log_write(f"[START] {exp_name} on {gpu} | dataset={config.get('dataset')} | tasks={task_desc}")
                        log_finished(f"job launch {exp_name}")

                        try:
                            os.remove(f"{jobs_folder}/{config_file}")
                        except FileNotFoundError:
                            pass

                        time.sleep(1)
                    else:
                        # Container died immediately. Track failures and skip after MAX_JOB_RETRIES.
                        fail_count = job_fail_counts.get(config_file, 0) + 1
                        job_fail_counts[config_file] = fail_count
                        if fail_count >= MAX_JOB_RETRIES:
                            log_write(f"!! Job {config_file} failed {fail_count} times. Skipping. !!")
                            try:
                                os.remove(f"{jobs_folder}/{config_file}")
                            except FileNotFoundError:
                                pass
                            job_fail_counts.pop(config_file, None)
                        else:
                            log_write(f"[WARN] Job start failed ({fail_count}/{MAX_JOB_RETRIES}). Will retry next loop.")
                            time.sleep(5)  # brief cooldown to let Docker/GPU settle
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
log_finished("all scheduled jobs")
print(f"\nAll {total_jobs} jobs finished!")
