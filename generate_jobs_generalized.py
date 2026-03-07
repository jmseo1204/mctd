import os
import json
import yaml
import argparse
from datetime import datetime
import copy
from pathlib import Path

def detect_frame_stack_from_ckpt(model_id, obs_dim, act_dim, downloaded_dir="outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval"):
    """
    Infer frame_stack from checkpoint weights.
    init_mlp.0.weight has shape (hidden, x_dim + k_embed_dim + external_cond_dim).
    out.weight has shape (x_dim, hidden).
    x_dim = frame_stack * (obs_dim + act_dim)
    => frame_stack = x_dim // (obs_dim + act_dim)
    """
    ckpt_path = Path(downloaded_dir) / model_id / "model.ckpt"
    if not ckpt_path.exists():
        return None

    def _parse_from_sd(sd):
        out_w = sd.get("diffusion_model.model.out.weight")
        if out_w is None:
            return None
        x_dim = out_w.shape[0]
        # Use data_mean shape as ground truth for unstacked_dim
        data_mean_w = sd.get("data_mean")
        if data_mean_w is not None:
            unstacked_dim = data_mean_w.shape[0]
        else:
            unstacked_dim = obs_dim + act_dim
        if unstacked_dim == 0 or x_dim % unstacked_dim != 0:
            return None
        fs = x_dim // unstacked_dim
        print(f"  [ckpt detect] x_dim={x_dim}, unstacked_dim={unstacked_dim} → frame_stack={fs}")
        return fs

    # Try direct torch import first
    try:
        import torch
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("state_dict", {})
        return _parse_from_sd(sd)
    except ImportError:
        pass
    except Exception as e:
        print(f"  [ckpt detect] direct torch failed: {e}")
        return None

    # Fallback: run via conda env subprocess
    try:
        import subprocess, json as _json
        script = (
            f"import torch, json; "
            f"sd = torch.load('{ckpt_path}', map_location='cpu', weights_only=False).get('state_dict', {{}}); "
            f"w = sd.get('diffusion_model.model.out.weight'); "
            f"dm = sd.get('data_mean'); "
            f"print(json.dumps({{'out': list(w.shape) if w is not None else None, 'dm': list(dm.shape) if dm is not None else None}}))"
        )
        result = subprocess.run(
            ["conda", "run", "-n", "diff_force_env", "python", "-c", script],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            info = _json.loads(result.stdout.strip())
            out_shape = info.get("out")
            dm_shape = info.get("dm")
            if out_shape:
                x_dim = out_shape[0]
                unstacked_dim = dm_shape[0] if dm_shape else (obs_dim + act_dim)
                if unstacked_dim > 0 and x_dim % unstacked_dim == 0:
                    fs = x_dim // unstacked_dim
                    print(f"  [ckpt detect via conda] x_dim={x_dim}, unstacked_dim={unstacked_dim} → frame_stack={fs}")
                    return fs
    except Exception as e:
        print(f"  [ckpt detect] conda fallback failed: {e}")
    return None


def detect_network_size_from_ckpt(model_id, downloaded_dir="outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval"):
    """
    Infer network_size (hidden dimension) from checkpoint weights.
    Checks transformer layer self_attn.in_proj_weight shape[1] for hidden dim.
    """
    ckpt_path = Path(downloaded_dir) / model_id / "model.ckpt"
    if not ckpt_path.exists():
        return None

    def _parse_from_sd(sd):
        # Try to find any attention layer to get hidden dimension
        # Key path: diffusion_model.model.transformer.layers.0.self_attn.in_proj_weight
        attn_key = "diffusion_model.model.transformer.layers.0.self_attn.in_proj_weight"
        attn_w = sd.get(attn_key)
        if attn_w is not None:
            # in_proj_weight shape: [3*hidden, hidden] (Q,K,V concatenated), so shape[1] = hidden
            hidden = attn_w.shape[1]
            print(f"  [ckpt detect] transformer layer 0 in_proj_weight shape {list(attn_w.shape)} → network_size={hidden}")

            # Also get dim_feedforward from linear1
            linear1_key = "diffusion_model.model.transformer.layers.0.linear1.weight"
            linear1_w = sd.get(linear1_key)
            if linear1_w is not None:
                dim_ff = linear1_w.shape[0]
                print(f"  [ckpt detect] transformer layer 0 linear1.weight shape {list(linear1_w.shape)} → dim_feedforward={dim_ff}")
                return {"network_size": hidden, "dim_feedforward": dim_ff}

            return {"network_size": hidden, "dim_feedforward": None}

        # Fallback: try linear1 from first transformer layer feedforward
        linear1_key = "diffusion_model.model.transformer.layers.0.linear1.weight"
        linear1_w = sd.get(linear1_key)
        if linear1_w is not None:
            hidden = linear1_w.shape[1]
            dim_ff = linear1_w.shape[0]
            print(f"  [ckpt detect] transformer layer 0 linear1.weight shape {list(linear1_w.shape)} → network_size={hidden}, dim_feedforward={dim_ff}")
            return {"network_size": hidden, "dim_feedforward": dim_ff}

        return None

    # Try direct torch import first
    try:
        import torch
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("state_dict", {})
        return _parse_from_sd(sd)
    except ImportError:
        pass
    except Exception as e:
        print(f"  [ckpt detect] direct torch failed: {e}")
        return None

    # Fallback: run via conda env subprocess
    try:
        import subprocess, json as _json
        script = (
            f"import torch, json; "
            f"sd = torch.load('{ckpt_path}', map_location='cpu', weights_only=False).get('state_dict', {{}}); "
            f"attn_w = sd.get('diffusion_model.model.transformer.layers.0.self_attn.in_proj_weight'); "
            f"linear1_w = sd.get('diffusion_model.model.transformer.layers.0.linear1.weight'); "
            f"print(json.dumps({{'attn': list(attn_w.shape) if attn_w is not None else None, 'linear1': list(linear1_w.shape) if linear1_w is not None else None}}))"
        )
        result = subprocess.run(
            ["conda", "run", "-n", "diff_force_env", "python", "-c", script],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            info = _json.loads(result.stdout.strip())
            attn_shape = info.get("attn")
            linear1_shape = info.get("linear1")

            if attn_shape and len(attn_shape) >= 2:
                hidden = attn_shape[1]
                dim_ff = linear1_shape[0] if linear1_shape and len(linear1_shape) >= 2 else None
                print(f"  [ckpt detect via conda] transformer layer 0 in_proj_weight shape {attn_shape} → network_size={hidden}")
                if dim_ff is not None:
                    print(f"  [ckpt detect via conda] transformer layer 0 linear1.weight shape {linear1_shape} → dim_feedforward={dim_ff}")
                return {"network_size": hidden, "dim_feedforward": dim_ff}
            elif linear1_shape and len(linear1_shape) >= 2:
                hidden = linear1_shape[1]
                dim_ff = linear1_shape[0]
                print(f"  [ckpt detect via conda] transformer layer 0 linear1.weight shape {linear1_shape} → network_size={hidden}, dim_feedforward={dim_ff}")
                return {"network_size": hidden, "dim_feedforward": dim_ff}
    except Exception as e:
        print(f"  [ckpt detect] conda fallback failed: {e}")

    return None


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
    parser.add_argument("--episode_len", type=int, default=None, help="Override episode_len (useful for offline/local runs without config.yaml)")
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
    actual_episode_len = args.episode_len or model_metadata.get('episode_len') or full_cfg['dataset'].get('episode_len', 50)
    actual_jump = model_metadata.get('jump') or full_cfg['dataset'].get('jump', 1)

    # Detect frame_stack from checkpoint weights (works even without wandb config)
    obs_dim = len(full_cfg['dataset'].get('observation_mean', [2]))
    act_dim = full_cfg['dataset'].get('action_dim', 8)
    detected_frame_stack = detect_frame_stack_from_ckpt(args.model_id, obs_dim, act_dim)
    actual_frame_stack = model_metadata.get('frame_stack') or detected_frame_stack or full_cfg['algorithm'].get('frame_stack', 10)

    # Detect network_size and dim_feedforward from checkpoint weights
    detected_arch = detect_network_size_from_ckpt(args.model_id)
    if isinstance(detected_arch, dict):
        actual_network_size = detected_arch.get("network_size") or full_cfg['algorithm'].get('diffusion', {}).get('architecture', {}).get('network_size', 256)
        actual_dim_feedforward = detected_arch.get("dim_feedforward") or full_cfg['algorithm'].get('diffusion', {}).get('architecture', {}).get('dim_feedforward', 1024)
    else:
        actual_network_size = full_cfg['algorithm'].get('diffusion', {}).get('architecture', {}).get('network_size', 256)
        actual_dim_feedforward = full_cfg['algorithm'].get('diffusion', {}).get('architecture', {}).get('dim_feedforward', 1024)

    actual_horizon_scale = args.horizon_scale if args.horizon_scale is not None else get_default_horizon_scale()
    actual_seq_div = full_cfg['algorithm'].get('sequence_dividing_factor', 3)

    # Validate plan_tokens divisibility: horizon // frame_stack must be divisible by sequence_dividing_factor
    horizon = int(actual_episode_len * actual_horizon_scale)
    plan_tokens = horizon // actual_frame_stack
    if plan_tokens == 0 or plan_tokens % actual_seq_div != 0:
        # Find the smallest valid horizon_scale that yields plan_tokens divisible by seq_div
        min_tokens = actual_seq_div  # at least seq_div tokens
        min_horizon = min_tokens * actual_frame_stack
        actual_horizon_scale = min_horizon / actual_episode_len
        horizon = int(actual_episode_len * actual_horizon_scale)
        plan_tokens = horizon // actual_frame_stack
        print(f"WARNING: original horizon_scale would give invalid plan_tokens. Adjusted horizon_scale to {actual_horizon_scale:.4f} (horizon={horizon}, plan_tokens={plan_tokens})")

    print(f"Final Plan: episode_len={actual_episode_len}, frame_stack={actual_frame_stack}, horizon_scale={actual_horizon_scale}, plan_tokens={plan_tokens}")

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
        "algorithm.diffusion.architecture.network_size": actual_network_size,
        "algorithm.diffusion.architecture.dim_feedforward": actual_dim_feedforward,
        "dataset.jump": actual_jump,
        "algorithm.horizon_scale": actual_horizon_scale,
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
