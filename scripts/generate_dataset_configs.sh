#!/usr/bin/env bash
# scripts/generate_dataset_configs.sh
#
# Auto-generates configurations/dataset/*.yaml for every OGBench dataset
# found in DATASET_DIR.  Run once after downloading new datasets.
#
# Output:
#   configurations/dataset/{name}.yaml   (e.g. antmaze_giant_navigate.yaml)
#   One file per train-split *.npz (val splits and *.tmp files are skipped).
#
# Usage:
#   bash scripts/generate_dataset_configs.sh            # dry-run preview
#   bash scripts/generate_dataset_configs.sh --write    # actually write files

set -euo pipefail

# ── Hardcoded dataset directory (edit here to change) ─────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$(dirname "$PROJECT_DIR")/ogbench_data"
OGBENCH_SRC="$(dirname "$PROJECT_DIR")/ogbench"   # path to ogbench source (for gymnasium registry)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_DIR="$PROJECT_DIR/configurations/dataset"

# Parse flags
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --write) DRY_RUN=false ;;
        --dry-run) DRY_RUN=true ;;
        --no-write) DRY_RUN=true ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Usage: $0 [--dry-run | --no-write]" >&2
            exit 1
            ;;
    esac
done

if $DRY_RUN; then
    echo "=== DRY RUN — pass --write to actually create files ==="
fi

echo "Dataset dir : $DATASET_DIR"
echo "Config dir  : $CONFIG_DIR"
echo ""

python3 - "$DATASET_DIR" "$CONFIG_DIR" "$DRY_RUN" "$OGBENCH_SRC" <<'PYEOF'
import numpy as np
import os
import sys
from pathlib import Path

DATASET_DIR  = Path(sys.argv[1])
CONFIG_DIR   = Path(sys.argv[2])
DRY_RUN      = sys.argv[3].lower() == "true"
OGBENCH_SRC  = sys.argv[4] if len(sys.argv) > 4 else None

# Ensure ogbench is importable for gymnasium registry lookups.
if OGBENCH_SRC and OGBENCH_SRC not in sys.path:
    sys.path.insert(0, OGBENCH_SRC)

# ── Constants ─────────────────────────────────────────────────────────────────
# extract_episode_ratio_from_sample: read from dataset_generate_config.yaml.
# episode_len = int(sample_length * extract_episode_ratio_from_sample),
#   then snapped DOWN to nearest SNAP_UNIT (= jump * frame_stack from
#   ckpt_df_planning.yaml) and clamped to < sample_length.
# Set snap_unit = 0 in ckpt_df_planning.yaml context to disable snapping.
import re as _re_yaml

def _read_yaml_scalar(path, key, cast=str):
    """Read a top-level scalar value from a YAML file without a YAML parser."""
    try:
        text = Path(path).read_text()
        m = _re_yaml.search(rf"^\s*{_re_yaml.escape(key)}\s*:\s*(\S+)", text, _re_yaml.MULTILINE)
        return cast(m.group(1)) if m else None
    except Exception:
        return None

_dataset_cfg_path = Path(sys.argv[2]) / "dataset_generate_config.yaml"
_ckpt_cfg_path    = Path(sys.argv[2]).parent / "algorithm" / "ckpt_df_planning.yaml"

EXTRACT_EPISODE_RATIO = _read_yaml_scalar(_dataset_cfg_path, "extract_episode_ratio_from_sample", float) or 0.5

_jump        = _read_yaml_scalar(_ckpt_cfg_path, "jump",        int)
_frame_stack = _read_yaml_scalar(_ckpt_cfg_path, "frame_stack", int)
SNAP_UNIT = (_jump * _frame_stack) if (_jump and _frame_stack) else 0

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_dataset_name(name):
    """
    'antmaze-giant-navigate-v0' → ('antmaze', 'giant',  'navigate')
    'cube-single-play-v0'       → ('cube',    'single', 'play')
    Returns (maze_type, size, task_type) or None if pattern not recognised.
    """
    TYPE_WORDS = {"navigate", "stitch", "explore", "play"}
    parts = name.rstrip("-v0").rstrip("-").split("-")
    # strip version suffix robustly
    if parts and parts[-1].startswith("v") and parts[-1][1:].isdigit():
        parts = parts[:-1]
    type_words_found = [p for p in parts if p in TYPE_WORDS]
    other_parts      = [p for p in parts if p not in TYPE_WORDS]
    if not type_words_found or len(other_parts) < 2:
        return None
    maze_type = other_parts[0]          # e.g. 'antmaze'
    size      = "_".join(other_parts[1:])  # e.g. 'giant' or 'large'
    task_type = type_words_found[0]     # e.g. 'navigate'
    return maze_type, size, task_type

def env_id_from_name(name):
    """
    'antmaze-giant-navigate-v0' → 'antmaze-giant-v0'
    'cube-single-play-v0'       → 'cube-single-v0'
    """
    TYPE_WORDS = {"navigate", "stitch", "explore", "play"}
    parts = name.split("-")
    kept = [p for p in parts if p not in TYPE_WORDS]
    return "-".join(kept)

def config_stem_from_name(name):
    """'antmaze-giant-navigate-v0' → 'antmaze_giant_navigate'"""
    # Remove -v0 (or -v1 etc.) suffix, then replace hyphens with underscores
    import re
    cleaned = re.sub(r"-v\d+$", "", name)
    return cleaned.replace("-", "_")

def get_num_tasks(env_id):
    """Read num_tasks by counting task_infos entries in the ogbench environment.
    Tries direct import first; falls back to Docker (mctd:0.1) if unavailable.
    """
    def _count_via_import():
        import gymnasium as _gym
        import ogbench as _ogbench  # ensure env registrations are loaded
        env = _gym.make(env_id)
        uw = env.unwrapped
        ti = uw.task_infos if hasattr(uw, 'task_infos') else env.task_infos
        count = len(ti)
        env.close()
        return count if count > 0 else None

    def _count_via_docker():
        import subprocess
        py_cmd = (
            "import gymnasium, ogbench\n"
            f"env = gymnasium.make('{env_id}')\n"
            "uw = env.unwrapped\n"
            "ti = uw.task_infos if hasattr(uw, 'task_infos') else env.task_infos\n"
            "print(len(ti))\n"
            "env.close()\n"
        )
        result = subprocess.run(
            ["docker", "run", "--rm", "mctd:0.1", "python3", "-c", py_cmd],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            val = result.stdout.strip().split()[-1]  # last token (skip Docker header lines)
            if val.isdigit():
                return int(val) if int(val) > 0 else None
        return None

    try:
        return _count_via_import()
    except Exception:
        pass
    try:
        return _count_via_docker()
    except Exception:
        return None

# ── Scan npz files ─────────────────────────────────────────────────────────
npz_files = sorted(DATASET_DIR.glob("*.npz"))
if not npz_files:
    print(f"No *.npz files found in {DATASET_DIR}")
    sys.exit(0)

print(f"Found {len(npz_files)} .npz file(s) in {DATASET_DIR}\n")

generated = []
skipped   = []
dry_run_preview = []

for npz_path in npz_files:
    stem = npz_path.stem  # 'antmaze-giant-navigate-v0'

    # Skip val splits
    if stem.endswith("-val"):
        print(f"[skip] {npz_path.name}  (val split)")
        continue

    parsed = parse_dataset_name(stem)
    if parsed is None:
        print(f"[skip] {npz_path.name}  (name not recognised)")
        skipped.append((str(npz_path), "name not recognised"))
        continue

    maze_type, size, task_type = parsed
    config_stem = config_stem_from_name(stem)
    config_path = CONFIG_DIR / f"{config_stem}.yaml"

    print(f"Processing: {npz_path.name}")
    print(f"  → {config_path.name}")

    try:
        data = np.load(str(npz_path), allow_pickle=False)
        obs       = data["observations"]
        terminals = data["terminals"]

        n_samples = len(obs)
        obs_dim   = obs.shape[1]

        # Infer sample_length from the first terminal index (all episodes are equal length).
        term_indices = np.where(terminals)[0]
        if len(term_indices) == 0:
            raise ValueError("No terminal transitions found; cannot infer sample_length.")
        sample_length = int(term_indices[0]) + 1

        # Observation normalization stats (over all training samples)
        obs_mean = np.mean(obs, axis=0).tolist()
        obs_std  = np.std(obs,  axis=0).tolist()

        # Reward stats: rewards are synthetic terminal signals (0 everywhere, 1 at
        # episode end).  reward_mean/std are stored for completeness but are unused
        # in both the dataset class and the algorithm (dead parameters).
        n_terminals = int(terminals.sum())
        reward_mean = float(n_terminals / n_samples)
        reward_std  = float((reward_mean * (1.0 - reward_mean)) ** 0.5)

        env_id = env_id_from_name(stem)

        # num_tasks: read from gymnasium registry (counts singletask-task{N} variants).
        num_tasks = get_num_tasks(env_id)
        num_tasks_str = str(num_tasks) if num_tasks is not None else "??  # could not detect; set manually"

        # ── Format output ──────────────────────────────────────────────────
        def fmt_list(vals, precision=8):
            return "[" + ", ".join(f"{v:.{precision}g}" for v in vals) + "]"

        raw_len = min(int(sample_length * EXTRACT_EPISODE_RATIO), sample_length - 1)
        episode_len = (raw_len // SNAP_UNIT) * SNAP_UNIT if SNAP_UNIT > 0 else raw_len

        lines = [
            f"defaults:",
            f"  - dataset_generate_config",
            f"",
            f"# ── [Group A: Dataset Loader] ─────────────────────────────────────────────",
            f"# Read directly by dataset.__init__() and env creation.",
            f"# Required at both train and eval time.",
            f'env_id: "{env_id}"',
            f'dataset: "{stem}"',
            f"save_dir: ~/.ogbench/data",
            f"num_tasks: {num_tasks_str}",
            f"# sample_length: frames per raw episode in the npz (auto-extracted at config generation time).",
            f"# episode_len = floor(int(sample_length * ratio) / snap_unit) * snap_unit",
            f"#   where snap_unit = jump * frame_stack (from ckpt_df_planning.yaml).",
            f"# Change sliding-window size via: dataset.extract_episode_ratio_from_sample=<ratio>",
            f"sample_length: {sample_length}",
            f"episode_len: {episode_len}  # ratio={EXTRACT_EPISODE_RATIO}, snap_unit={SNAP_UNIT}",
            f"",
            f"# ── [Group B: Normalization Stats] ────────────────────────────────────────",
            f"# train time: interpolated into algorithm cfg via train_df_planning.yaml",
            f"#             (${'{'}dataset.observation_mean{'}'} etc.) and saved to ckpt training_hparams.",
            f"# eval time:  restored from ckpt training_hparams; yaml value is NOT used.",
            f"# Computed from {stem} training data",
            f"# ({n_samples} samples, obs_dim={obs_dim}).",
            f"observation_mean: {fmt_list(obs_mean)}",
            f"observation_std:  {fmt_list(obs_std)}",
            f"# action_mean/std: empty → action_dim=0.",
            f"# The diffusion model generates observations only; actions are computed at",
            f"# runtime by DQL (antmaze) or PID (pointmaze) controller.",
            f"action_mean: []",
            f"action_std:  []",
            f"reward_mean: {reward_mean:.6g}  # = n_terminals/n_samples (synthetic terminal reward)",
            f"reward_std:  {reward_std:.6g}",
        ]
        content = "\n".join(lines) + "\n"

        if DRY_RUN:
            dry_run_preview.append((config_path, content))
            print(f"  [dry-run] would write {config_path}")
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(config_path), "w") as f:
                f.write(content)
            generated.append(str(config_path))
            print(f"  [written] {config_path}")

        print(f"  obs_dim={obs_dim}, sample_length={sample_length}, n_samples={n_samples}, "
              f"n_episodes={n_terminals}, reward_mean={reward_mean:.5f}")

    except Exception as e:
        import traceback
        print(f"  [error] {e}")
        traceback.print_exc()
        skipped.append((str(npz_path), str(e)))

    print()

# ── Summary ────────────────────────────────────────────────────────────────
print("=" * 60)
if DRY_RUN:
    print(f"DRY RUN: {len(dry_run_preview)} config(s) would be written.")
    print("Re-run with --write to create files.")
    if dry_run_preview:
        print("\n--- Preview of first config ---")
        print(dry_run_preview[0][1])
else:
    print(f"Generated {len(generated)} config(s).")
    for p in generated:
        print(f"  {p}")
if skipped:
    print(f"\nSkipped {len(skipped)} file(s):")
    for path, reason in skipped:
        print(f"  {path}: {reason}")
PYEOF
