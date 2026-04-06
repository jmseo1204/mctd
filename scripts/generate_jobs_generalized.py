import os
import json
import yaml
import argparse
from datetime import datetime
import copy
from pathlib import Path
from project_config import DOCKER_USER as _DOCKER_USER, WANDB_ENTITY as _WANDB_ENTITY, \
    DOCKER_IMAGE as _DOCKER_IMAGE, WANDB_PROJECT as _WANDB_PROJECT

def detect_frame_stack_from_ckpt(model_id, obs_dim, act_dim, downloaded_dir=f"/home/{_DOCKER_USER}/mctd/outputs/downloaded/{_WANDB_ENTITY}/{_WANDB_PROJECT}"):
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

    # Fallback: run via Docker (mctd:0.1 has torch; host may not)
    try:
        import subprocess, json as _json
        real_path = str(ckpt_path.resolve())
        mount_dir = str(ckpt_path.resolve().parent)
        script = (
            f"import torch, json; "
            f"sd = torch.load('{real_path}', map_location='cpu', weights_only=False).get('state_dict', {{}}); "
            f"w = sd.get('diffusion_model.model.out.weight'); "
            f"dm = sd.get('data_mean'); "
            f"print(json.dumps({{'out': list(w.shape) if w is not None else None, 'dm': list(dm.shape) if dm is not None else None}}))"
        )
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3",
             "-v", f"{mount_dir}:{mount_dir}:ro",
             _DOCKER_IMAGE, "-c", script],
            capture_output=True, text=True, timeout=120
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
                    print(f"  [ckpt detect via docker] x_dim={x_dim}, unstacked_dim={unstacked_dim} → frame_stack={fs}")
                    return fs
    except Exception as e:
        print(f"  [ckpt detect] docker fallback failed: {e}")
    return None


def detect_network_size_from_ckpt(model_id, downloaded_dir=f"/home/{_DOCKER_USER}/mctd/outputs/downloaded/{_WANDB_ENTITY}/{_WANDB_PROJECT}"):
    """
    Infer network_size (hidden dimension) from checkpoint weights.
    Checks transformer layer self_attn.in_proj_weight shape[1] for hidden dim.
    """
    ckpt_path = Path(downloaded_dir) / model_id / "model.ckpt"
    if not ckpt_path.exists():
        return None

    def _parse_from_sd(sd):
        # Count number of transformer layers
        layer_indices = set()
        for k in sd.keys():
            if "diffusion_model.model.transformer.layers." in k:
                try:
                    idx = int(k.split("diffusion_model.model.transformer.layers.")[1].split(".")[0])
                    layer_indices.add(idx)
                except (IndexError, ValueError):
                    pass
        num_layers = max(layer_indices) + 1 if layer_indices else None
        if num_layers is not None:
            print(f"  [ckpt detect] found transformer layers {sorted(layer_indices)} → num_layers={num_layers}")

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
                return {"network_size": hidden, "dim_feedforward": dim_ff, "num_layers": num_layers}

            return {"network_size": hidden, "dim_feedforward": None, "num_layers": num_layers}

        # Fallback: try linear1 from first transformer layer feedforward
        linear1_key = "diffusion_model.model.transformer.layers.0.linear1.weight"
        linear1_w = sd.get(linear1_key)
        if linear1_w is not None:
            hidden = linear1_w.shape[1]
            dim_ff = linear1_w.shape[0]
            print(f"  [ckpt detect] transformer layer 0 linear1.weight shape {list(linear1_w.shape)} → network_size={hidden}, dim_feedforward={dim_ff}")
            return {"network_size": hidden, "dim_feedforward": dim_ff, "num_layers": num_layers}

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

    # Fallback: run via Docker (mctd:0.1 has torch; host may not)
    # Combines arch weights + num_layers into a single Docker call.
    try:
        import subprocess, json as _json
        real_path = str(ckpt_path.resolve())
        mount_dir = str(ckpt_path.resolve().parent)
        script = (
            f"import torch, json; "
            f"sd = torch.load('{real_path}', map_location='cpu', weights_only=False).get('state_dict', {{}}); "
            f"attn_w = sd.get('diffusion_model.model.transformer.layers.0.self_attn.in_proj_weight'); "
            f"linear1_w = sd.get('diffusion_model.model.transformer.layers.0.linear1.weight'); "
            f"idxs = set(int(k.split('transformer.layers.')[1].split('.')[0]) for k in sd if 'transformer.layers.' in k) if any('transformer.layers.' in k for k in sd) else set(); "
            f"num_layers = max(idxs)+1 if idxs else 0; "
            f"print(json.dumps({{'attn': list(attn_w.shape) if attn_w is not None else None, 'linear1': list(linear1_w.shape) if linear1_w is not None else None, 'num_layers': num_layers}}))"
        )
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3",
             "-v", f"{mount_dir}:{mount_dir}:ro",
             _DOCKER_IMAGE, "-c", script],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            info = _json.loads(result.stdout.strip())
            attn_shape = info.get("attn")
            linear1_shape = info.get("linear1")
            num_layers = info.get("num_layers") or None
            if num_layers:
                print(f"  [ckpt detect via docker] num_layers={num_layers}")

            if attn_shape and len(attn_shape) >= 2:
                hidden = attn_shape[1]
                dim_ff = linear1_shape[0] if linear1_shape and len(linear1_shape) >= 2 else None
                print(f"  [ckpt detect via docker] transformer layer 0 in_proj_weight shape {attn_shape} → network_size={hidden}")
                if dim_ff is not None:
                    print(f"  [ckpt detect via docker] transformer layer 0 linear1.weight shape {linear1_shape} → dim_feedforward={dim_ff}")
                return {"network_size": hidden, "dim_feedforward": dim_ff, "num_layers": num_layers}
            elif linear1_shape and len(linear1_shape) >= 2:
                hidden = linear1_shape[1]
                dim_ff = linear1_shape[0]
                print(f"  [ckpt detect via docker] transformer layer 0 linear1.weight shape {linear1_shape} → network_size={hidden}, dim_feedforward={dim_ff}")
                return {"network_size": hidden, "dim_feedforward": dim_ff, "num_layers": num_layers}
    except Exception as e:
        print(f"  [ckpt detect] docker fallback failed: {e}")

    return None


def find_local_training_config(model_id, downloaded_dir=f"/home/{_DOCKER_USER}/mctd/outputs/downloaded/{_WANDB_ENTITY}/{_WANDB_PROJECT}"):
    """
    Find training_config.yaml saved by train.sh alongside the checkpoint.
    Written in plain-YAML format; extract_from_config() handles it without changes.
    """
    config_path = Path(downloaded_dir) / model_id / "training_config.yaml"
    if config_path.exists():
        return config_path
    return None


def find_config_yaml(model_id, outputs_root=None):
    if outputs_root is None:
        outputs_root = f"/home/{_DOCKER_USER}/mctd/outputs"
    """
    Search for config.yaml in WANDB run directories matching the model_id.

    Priority:
    1. WandB training run dir: *-{model_id}/files/config.yaml  (run IS the training run)
    2. Hydra eval run dir: .hydra/config.yaml where load == model_id
       → picks the OLDEST such file (least contaminated by later config changes)
    """
    outputs_path = Path(outputs_root)
    if not outputs_path.exists():
        return None

    # 1. Training run: WandB run directory ends with model_id
    pattern = f"*-{model_id}"
    matches = list(outputs_path.glob(f"**/{pattern}/files/config.yaml"))
    if matches:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0]

    # 2. Eval run: .hydra/config.yaml where `load: {model_id}`
    hydra_matches = []
    for hydra_cfg in outputs_path.glob("**/.hydra/config.yaml"):
        try:
            with open(hydra_cfg, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("load") == model_id:
                hydra_matches.append(hydra_cfg)
        except Exception:
            continue
    if hydra_matches:
        # Oldest first — most likely to reflect the original training config
        hydra_matches.sort(key=lambda x: x.stat().st_mtime)
        chosen = hydra_matches[0]
        print(f"  [config detect] Training config not found for {model_id}; "
              f"using oldest eval .hydra/config.yaml: {chosen}")
        return chosen
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
        # Model training-dependent architecture params (must match training config)
        'causal': get_val(['algorithm', 'causal']),
        'scheduling_matrix': get_val(['algorithm', 'scheduling_matrix']),
        'padding_mode': get_val(['algorithm', 'padding_mode']),
        'context_frames': get_val(['algorithm', 'context_frames']),
        'obs_dim_indices': get_val(['algorithm', 'obs_dim_indices']),
        'pos_dim_indices': get_val(['algorithm', 'pos_dim_indices']),
        'attn_heads': get_val(['algorithm', 'diffusion', 'architecture', 'attn_heads']),
        'network_size': get_val(['algorithm', 'diffusion', 'architecture', 'network_size']),
        'num_layers': get_val(['algorithm', 'diffusion', 'architecture', 'num_layers']),
        'dim_feedforward': get_val(['algorithm', 'diffusion', 'architecture', 'dim_feedforward']),
        # Training dataset identity (used by eval.sh to pick correct dataset)
        'dataset_config': get_val(['dataset', 'config']),
    }

    # Fallbacks — try alternative paths when primary lookup fails
    if metadata['episode_len'] is None:
        metadata['episode_len'] = get_val(['dataset', 'episode_len'])
    if metadata['jump'] is None:
        metadata['jump'] = get_val(['jump'])  # Sometimes at root
    if metadata['frame_stack'] is None:
        # training_config.yaml stores frame_stack directly under algorithm
        metadata['frame_stack'] = get_val(['algorithm', 'frame_stack'])

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
        # Fallback: strip legacy 'og_' prefix if file not found
        if not dataset_path.exists() and dataset_name.startswith("og_"):
            stripped = dataset_name[3:]
            fallback_path = Path(f"configurations/dataset/{stripped}.yaml")
            if fallback_path.exists():
                print(f"  [load_full_config] Remapped legacy dataset name: {dataset_name} → {stripped}")
                dataset_name = stripped
                dataset_path = fallback_path
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

def load_training_hparams_from_ckpt(model_id, downloaded_dir=f"/home/{_DOCKER_USER}/mctd/outputs/downloaded/{_WANDB_ENTITY}/{_WANDB_PROJECT}"):
    """Load training_hparams saved by df_base.on_save_checkpoint from the checkpoint file.

    New checkpoints (trained after the save_hyperparameters refactor) embed all arch
    and behavioral params directly.  Returns {} for old checkpoints that predate this.
    Falls back to Docker if torch is not available on the host.
    """
    ckpt_path = Path(downloaded_dir) / model_id / "model.ckpt"
    if not ckpt_path.exists():
        return {}

    # Direct load (host has torch)
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        hparams = ckpt.get("training_hparams", {})
        if hparams:
            print(f"  [ckpt hparams] Loaded training_hparams from checkpoint: {list(hparams.keys())}")
        return hparams
    except ImportError:
        pass
    except Exception as e:
        print(f"  [ckpt hparams] Direct load failed: {e}")
        return {}

    # Docker fallback (host lacks torch)
    try:
        import subprocess, json as _json
        real_path = str(ckpt_path.resolve())
        mount_dir = str(ckpt_path.resolve().parent)
        script = (
            "import torch, json; "
            f"ckpt = torch.load('{real_path}', map_location='cpu', weights_only=False); "
            "h = ckpt.get('training_hparams', {}); "
            "print(json.dumps(h))"
        )
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3",
             "-v", f"{mount_dir}:{mount_dir}:ro",
             _DOCKER_IMAGE, "-c", script],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            hparams = _json.loads(result.stdout.strip())
            if hparams:
                print(f"  [ckpt hparams via docker] Loaded training_hparams: {list(hparams.keys())}")
            return hparams
    except Exception as e:
        print(f"  [ckpt hparams] Docker fallback failed: {e}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation jobs JSON files.")
    parser.add_argument("--dataset", required=True, help="Dataset config name (e.g., antmaze_giant_stitch)")
    parser.add_argument("--model_id", required=True, help="WandB model ID (e.g., en1ddvu7)")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks to generate")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds per task")
    parser.add_argument("--start_task_id", type=int, default=1, help="Starting task index (1-based)")
    parser.add_argument("--horizon_scale", type=float, default=None, help="Override Multiplier")
    parser.add_argument("--episode_len", type=int, default=None, help="Override episode_len (useful for offline/local runs without config.yaml)")
    parser.add_argument("--outputs_root", default=f"/home/{_DOCKER_USER}/mctd/outputs", help="Root directory of outputs/wandb logs")

    args = parser.parse_args()

    # 1. Load current project YAMLs as foundation
    full_cfg = load_full_config(args.dataset)
    if full_cfg is None: return

    print(f"--- Meta Search for Model ID: {args.model_id} ---")

    # Legacy-path variables (only populated when checkpoint has no training_hparams)
    actual_obs_dim_indices: list = []
    actual_network_size = None
    actual_dim_feedforward = None
    actual_num_layers = None
    actual_causal = None
    actual_scheduling_matrix = None
    actual_attn_heads = None

    # 2. Try new path: training_hparams embedded in checkpoint
    ckpt_hparams = load_training_hparams_from_ckpt(args.model_id)

    if ckpt_hparams:
        # ── New path: checkpoint carries all training params ──────────────────
        # episode_len in ckpt is already effective (raw // jump), saved by on_save_checkpoint.
        actual_jump = ckpt_hparams.get('jump', 1)
        actual_episode_len = (
            args.episode_len
            or ckpt_hparams.get('episode_len')
            or full_cfg['dataset'].get('episode_len', 50) // actual_jump
        )
        actual_frame_stack = ckpt_hparams.get('frame_stack', 10)
        actual_obs_dim_indices = ckpt_hparams.get('obs_dim_indices')
        actual_causal = ckpt_hparams.get('causal')
        actual_scheduling_matrix = ckpt_hparams.get('scheduling_matrix')
        actual_padding_mode = ckpt_hparams.get('padding_mode')
        actual_context_frames = ckpt_hparams.get('context_frames')
        actual_frame_skip = ckpt_hparams.get('frame_skip')
        actual_external_cond_dim = ckpt_hparams.get('external_cond_dim')
        actual_uncertainty_scale = ckpt_hparams.get('uncertainty_scale')
        _arch = (ckpt_hparams.get('diffusion') or {}).get('architecture', {})
        actual_network_size = _arch.get('network_size')
        actual_dim_feedforward = _arch.get('dim_feedforward')
        actual_num_layers = _arch.get('num_layers')
        actual_attn_heads = _arch.get('attn_heads')
        print(
            f"  [ckpt hparams] Using embedded training_hparams: "
            f"episode_len={actual_episode_len}, frame_stack={actual_frame_stack}, "
            f"causal={actual_causal}, "
            f"scheduling_matrix={actual_scheduling_matrix}"
        )
    else:
        # ── Legacy path: detect params from weights + wandb/local config ─────
        config_path = find_config_yaml(args.model_id, args.outputs_root)

        model_metadata = {}
        if config_path:
            print(f"Found WandB config for model at: {config_path}")
            model_metadata = extract_from_config(config_path)
            print(f"Model-specific metadata (from WandB config): {model_metadata}")
        else:
            local_config = find_local_training_config(args.model_id)
            if local_config:
                print(f"Found local training config at: {local_config}")
                model_metadata = extract_from_config(local_config)
                print(f"Model-specific metadata (from training_config.yaml): {model_metadata}")

        # If training config names a different dataset, reload with that dataset
        detected_dataset_config = model_metadata.get('dataset_config')
        # Normalize legacy 'og_' prefix so comparison is stable
        if detected_dataset_config and detected_dataset_config.startswith("og_"):
            _stripped = detected_dataset_config[3:]
            if Path(f"configurations/dataset/{_stripped}.yaml").exists():
                print(f"  [config detect] Remapped legacy dataset name: {detected_dataset_config} → {_stripped}")
                detected_dataset_config = _stripped
        if detected_dataset_config and detected_dataset_config != args.dataset:
            print(f"  [config detect] Training dataset '{detected_dataset_config}' differs from arg '{args.dataset}'. "
                  f"Reloading config with training dataset.")
            reloaded = load_full_config(detected_dataset_config)
            if reloaded:
                full_cfg = reloaded
                args.dataset = detected_dataset_config
            else:
                print(f"  [config detect] WARNING: Could not load '{detected_dataset_config}'. Keeping '{args.dataset}'.")

        # Determine jump first (needed to divide dataset raw episode_len).
        # metadata already stores the effective (divided) episode_len, so no post-processing there.
        actual_jump = model_metadata.get('jump') or 1
        dataset_episode_len = full_cfg['dataset'].get('episode_len')
        actual_episode_len = (
            args.episode_len
            or (dataset_episode_len // actual_jump if dataset_episode_len else None)
            or model_metadata.get('episode_len')
            or 50
        )

        # obs_dim_indices: prefer from training_config.yaml (set after our fix), else infer from dataset
        actual_obs_dim_indices = (
            model_metadata.get('obs_dim_indices')
            or list(range(len(full_cfg['dataset'].get('observation_mean', [0, 0]))))
        )
        obs_dim = len(actual_obs_dim_indices)
        act_dim = full_cfg['dataset'].get('action_dim', 8)
        detected_frame_stack = detect_frame_stack_from_ckpt(args.model_id, obs_dim, act_dim)
        actual_frame_stack = detected_frame_stack or model_metadata.get('frame_stack') or full_cfg['algorithm'].get('frame_stack', 10)

        detected_arch = detect_network_size_from_ckpt(args.model_id)
        arch_cfg = full_cfg['algorithm'].get('diffusion', {}).get('architecture', {})
        if isinstance(detected_arch, dict):
            actual_network_size = detected_arch.get("network_size") or model_metadata.get('network_size') or arch_cfg.get('network_size', 64)
            actual_dim_feedforward = detected_arch.get("dim_feedforward") or model_metadata.get('dim_feedforward') or arch_cfg.get('dim_feedforward', 256)
            actual_num_layers = detected_arch.get("num_layers") or model_metadata.get('num_layers') or arch_cfg.get('num_layers', 6)
        else:
            actual_network_size = model_metadata.get('network_size') or arch_cfg.get('network_size', 64)
            actual_dim_feedforward = model_metadata.get('dim_feedforward') or arch_cfg.get('dim_feedforward', 256)
            actual_num_layers = model_metadata.get('num_layers') or arch_cfg.get('num_layers', 6)

        actual_causal = model_metadata.get('causal')
        actual_scheduling_matrix = model_metadata.get('scheduling_matrix')
        actual_attn_heads = model_metadata.get('attn_heads')
        actual_padding_mode = model_metadata.get('padding_mode')
        actual_context_frames = model_metadata.get('context_frames')
        actual_frame_skip = model_metadata.get('frame_skip')
        actual_external_cond_dim = None  # not recoverable from legacy configs
        actual_uncertainty_scale = None  # not recoverable from legacy configs
        missing = [k for k, v in [('causal', actual_causal), ('scheduling_matrix', actual_scheduling_matrix), ('attn_heads', actual_attn_heads)] if v is None]
        if missing:
            print(f"WARNING: Could not detect {missing} from saved wandb config. "
                  f"These will use current yaml defaults, which may be wrong for this checkpoint.")
        else:
            print(f"Detected training config: causal={actual_causal}, scheduling_matrix={actual_scheduling_matrix}, attn_heads={actual_attn_heads}")

    # 3. Validate plan_tokens divisibility
    actual_horizon_scale = args.horizon_scale if args.horizon_scale is not None else get_default_horizon_scale()
    actual_seq_div = full_cfg['algorithm'].get('sequence_dividing_factor', 3)
    horizon = int(actual_episode_len * actual_horizon_scale)
    plan_tokens = horizon // actual_frame_stack
    if plan_tokens == 0 or plan_tokens % actual_seq_div != 0:
        min_horizon = actual_seq_div * actual_frame_stack
        actual_horizon_scale = min_horizon / actual_episode_len
        horizon = int(actual_episode_len * actual_horizon_scale)
        plan_tokens = horizon // actual_frame_stack
        print(f"WARNING: original horizon_scale would give invalid plan_tokens. "
              f"Adjusted horizon_scale to {actual_horizon_scale:.4f} (horizon={horizon}, plan_tokens={plan_tokens})")

    print(f"Final Plan: episode_len={actual_episode_len}, frame_stack={actual_frame_stack}, "
          f"horizon_scale={actual_horizon_scale}, plan_tokens={plan_tokens}")

    # 4. Build job config
    basic_job_config = {
        "wandb.entity": _WANDB_ENTITY,
        "wandb.project": _WANDB_PROJECT,
        "wandb.group": f"EVAL-{args.model_id}",
        "experiment": "base_pytorch",
        "algorithm": "df_planning",
        "load": args.model_id,
        "dataset": args.dataset,
    }

    basic_job_config.update({
        "dataset.jump": actual_jump,
        "algorithm.horizon_scale": actual_horizon_scale,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "experiment.validation.precision": 32,
        "experiment.validation.inference_mode": False,
        "experiment.validation.limit_batch": 1,
    })

    # Always embed all resolved arch params so exp_base.py can skip reloading the ckpt.
    arch_overrides = {
        "dataset.episode_len": actual_episode_len,
        "algorithm.jump": actual_jump,
        "algorithm.frame_stack": actual_frame_stack,
        "algorithm.obs_dim_indices": actual_obs_dim_indices,
        "algorithm.diffusion.architecture.network_size": actual_network_size,
        "algorithm.diffusion.architecture.dim_feedforward": actual_dim_feedforward,
        "algorithm.diffusion.architecture.num_layers": actual_num_layers,
    }
    for key, val in [
        ("algorithm.causal",                                actual_causal),
        ("algorithm.scheduling_matrix",                     actual_scheduling_matrix),
        ("algorithm.padding_mode",                          actual_padding_mode),
        ("algorithm.context_frames",                        actual_context_frames),
        ("algorithm.frame_skip",                            actual_frame_skip),
        ("algorithm.external_cond_dim",                     actual_external_cond_dim),
        ("algorithm.uncertainty_scale",                     actual_uncertainty_scale),
        ("algorithm.diffusion.architecture.attn_heads",     actual_attn_heads),
    ]:
        if val is not None:
            arch_overrides[key] = val
    basic_job_config.update(arch_overrides)

    # 5. Generate Jobs
    jobs_folder = "jobs"
    if not os.path.exists(jobs_folder):
        os.makedirs(jobs_folder)

    total_tasks_num = full_cfg['dataset'].get('num_tasks')
    start_task_id = args.start_task_id
    count = 0
    for i in range(args.num_tasks):
        if total_tasks_num:
            task_id = ((start_task_id - 1 + i) % total_tasks_num) + 1
        else:
            task_id = start_task_id + i
        for seed in range(args.num_seeds):
            job_cfg = copy.deepcopy(basic_job_config)
            job_cfg["algorithm.interaction_seed"] = seed
            job_cfg["algorithm.task_id"] = task_id
            job_cfg["+name"] = f"EVAL_{args.model_id}_T{task_id}_S{seed}"

            filename = f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json"
            with open(filename, "w") as f:
                json.dump(job_cfg, f, indent=4)
            count += 1

    print(f"Successfully generated {count} jobs in '{jobs_folder}/' folder.")

if __name__ == "__main__":
    main()
