import os
import json
import yaml
import argparse
from datetime import datetime
import copy
from pathlib import Path

def find_config_yaml(model_id, outputs_root="/home/jmseo1204/mctd_outputs"):
    """
    Search for config.yaml in WANDB run directories matching the model_id.
    """
    outputs_path = Path(outputs_root)
    if not outputs_path.exists():
        return None
    
    # Search for directories that end with the model_id
    pattern = f"*-{model_id}"
    matches = list(outputs_path.glob(f"**/{pattern}/files/config.yaml"))
    
    if matches:
        # Return the most recent one if multiple matches
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0]
    return None

def get_default_horizon_scale(config_path="configurations/algorithm/df_planning.yaml"):
    """
    Load horizon_scale from the default algorithm config file.
    """
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('horizon_scale', 0.9) # default fallback
    except Exception:
        return 0.9

def extract_from_config(config_path):
    """
    Extract episode_len, frame_stack, jump from wandb config.yaml.
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # WandB config.yaml format: { 'key': { 'value': ... } }
    def get_val(path_list):
        curr = data
        for p in path_list:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
                if isinstance(curr, dict) and 'value' in curr:
                    curr = curr['value']
            else:
                return None
        return curr

    def resolve_val(val):
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            # Simple interpolation like ${dataset.episode_len}
            path = val[2:-1].split(".")
            return get_val(path)
        return val

    metadata = {
        'episode_len': resolve_val(get_val(['algorithm', 'episode_len'])),
        'frame_stack': resolve_val(get_val(['algorithm', 'frame_stack'])),
        'jump': resolve_val(get_val(['dataset', 'jump'])),
        'frame_skip': resolve_val(get_val(['algorithm', 'frame_skip'])),
    }
    
    # Fallbacks
    if metadata['episode_len'] is None:
        metadata['episode_len'] = get_val(['dataset', 'episode_len'])
    if metadata['jump'] is None:
        metadata['jump'] = get_val(['jump']) # Sometimes it is at root
    
    return metadata

def resolve_interpolations(config, dataset_cfg):
    """
    Manually resolve ${dataset.xxx} interpolations in the config dictionary.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            config[k] = resolve_interpolations(v, dataset_cfg)
    elif isinstance(config, list):
        for i in range(len(config)):
            config[i] = resolve_interpolations(config[i], dataset_cfg)
    elif isinstance(config, str) and config.startswith("${dataset."):
        # Extract the key name, e.g., ${dataset.episode_len} -> episode_len
        key_name = config.replace("${dataset.", "").replace("}", "")
        return dataset_cfg.get(key_name, config)
    return config

def load_full_config(dataset_name, algo_name="df_planning"):
    """
    Load the full config using standard yaml and manually resolve dataset interpolations.
    """
    try:
        import yaml
        dataset_path = Path(f"configurations/dataset/{dataset_name}.yaml")
        algo_path = Path(f"configurations/algorithm/{algo_name}.yaml")
        
        with open(dataset_path, "r") as f:
            ds_cfg = yaml.safe_load(f)
        with open(algo_path, "r") as f:
            algo_cfg = yaml.safe_load(f)
            
        # Manually resolve interpolations
        resolved_algo = resolve_interpolations(algo_cfg, ds_cfg)
        
        return {
            "dataset": ds_cfg,
            "algorithm": resolved_algo
        }
    except Exception as e:
        print(f"Error loading configs: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation jobs JSON files.")
    parser.add_argument("--dataset", required=True, help="Dataset config name (e.g., og_antmaze_giant_stitch)")
    parser.add_argument("--model_id", required=True, help="WandB model ID (e.g., en1ddvu7)")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks (1 to N)")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds per task")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repeats per seed/task (robustness)")
    parser.add_argument("--horizon_scale", type=float, default=None, help="Override Multiplier")
    parser.add_argument("--outputs_root", default="/home/jmseo1204/mctd_outputs", help="Root directory of outputs/wandb logs")
    
    args = parser.parse_args()

    # 1. Load current project YAMLs as foundation
    full_cfg = load_full_config(args.dataset)
    if full_cfg is None: return

    # 2. Extract model-specific metadata from checkpoint config (if exists)
    print(f"--- Meta Search for Model ID: {args.model_id} ---")
    config_path = find_config_yaml(args.model_id, args.outputs_root)
    
    model_metadata = {}
    if config_path:
        print(f"Found saved config for model at: {config_path}")
        model_metadata = extract_from_config(config_path)
        print(f"Model-specific metadata (from ckpt): {model_metadata}")

    # 3. Consolidate Config (Prioritize model_metadata for training consistency)
    actual_episode_len = model_metadata.get('episode_len') or full_cfg['dataset'].get('episode_len', 50)
    actual_frame_stack = model_metadata.get('frame_stack') or full_cfg['algorithm'].get('frame_stack', 10)
    actual_jump = model_metadata.get('jump') or full_cfg['dataset'].get('jump', 1)
    
    print(f"Final Plan: episode_len={actual_episode_len}, frame_stack={actual_frame_stack}")

    # 4. Build Minimal Basic Config (Let Hydra load YAMLs inside the container)
    basic_job_config = {
        "wandb.entity": "jmseo1204-seoul-national-university",
        "wandb.project": "mctd_eval",
        "wandb.group": f"EVAL-{args.model_id}",
        "experiment": "exp_planning",
        "algorithm": "df_planning", # This tells Hydra to load configurations/algorithm/df_planning.yaml
        "load": args.model_id,
        "dataset": args.dataset,    # This tells Hydra to load configurations/dataset/[dataset].yaml
    }

    # Apply strictly necessary metadata and overrides
    basic_job_config.update({
        "dataset.episode_len": actual_episode_len,
        "algorithm.frame_stack": actual_frame_stack,
        "dataset.jump": actual_jump,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
    })

    # 5. Generate Jobs
    jobs_folder = "jobs"
    if not os.path.exists(jobs_folder):
        os.makedirs(jobs_folder)
        
    count = 0
    for task_id in range(1, args.num_tasks + 1):
        for seed in range(args.num_seeds):
            for r in range(args.num_repeats):
                job_cfg = copy.deepcopy(basic_job_config)
                job_cfg["experiment.validation.seed"] = seed
                job_cfg["algorithm.task_id"] = task_id
                job_cfg["+name"] = f"EVAL_{args.model_id}_T{task_id}_S{seed}_R{r}"
                
                filename = f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json"
                with open(filename, "w") as f:
                    json.dump(job_cfg, f, indent=4)
                count += 1
                
    print(f"Successfully generated {count} jobs in '{jobs_folder}/' folder.")

if __name__ == "__main__":
    main()
