#!/usr/bin/env bash
# scripts/generate_dataset_configs.sh
#
# Auto-generates configurations/dataset/og_*.yaml for every OGBench dataset
# found in DATASET_DIR.  Run once after downloading new datasets.
#
# Output:
#   configurations/dataset/og_{name}.yaml   (e.g. og_antmaze_giant_navigate.yaml)
#   One file per train-split *.npz (val splits and *.tmp files are skipped).
#
# Usage:
#   bash scripts/generate_dataset_configs.sh            # dry-run preview
#   bash scripts/generate_dataset_configs.sh --write    # actually write files

set -euo pipefail

# ── Hardcoded dataset directory (edit here to change) ─────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$(dirname "$PROJECT_DIR")/ogbench_data"
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_DIR="$PROJECT_DIR/configurations/dataset"

# Parse flags
DRY_RUN=true
for arg in "$@"; do
    case "$arg" in
        --write) DRY_RUN=false ;;
        --dry-run) DRY_RUN=true ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Usage: $0 [--write | --dry-run]" >&2
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

python3 - "$DATASET_DIR" "$CONFIG_DIR" "$DRY_RUN" <<'PYEOF'
import numpy as np
import os
import sys
from pathlib import Path

DATASET_DIR = Path(sys.argv[1])
CONFIG_DIR  = Path(sys.argv[2])
DRY_RUN     = sys.argv[3].lower() == "true"

# ── Hardcoded metadata tables ─────────────────────────────────────────────────
# episode_len: the cfg value read by OGAntMazeOfflineRLDataset (raw, before /jump).
# Equals the intended max trajectory length.  The npz stores up to 2*episode_len+1
# frames per trajectory for navigate (sliding-window format).
EPISODE_LEN_TABLE = {
    ("antmaze", "giant",    "navigate"): 1000,
    ("antmaze", "large",    "navigate"):  500,
    ("antmaze", "medium",   "navigate"):  500,
    ("antmaze", "teleport", "navigate"):  500,
    ("antmaze", "giant",    "stitch"):    100,
    ("antmaze", "large",    "stitch"):    100,
    ("antmaze", "medium",   "stitch"):    100,
    ("antmaze", "teleport", "stitch"):    100,
    ("antmaze", "giant",    "explore"):   250,
    ("antmaze", "large",    "explore"):   250,
    ("antmaze", "medium",   "explore"):   250,
    ("antmaze", "teleport", "explore"):   250,
}
# num_tasks: all OGBench antmaze variants use 5 tasks.
NUM_TASKS = 5
# gamma: always 1.0 (sparse terminal reward; effectively dead parameter in algo).
GAMMA = 1.0

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_dataset_name(name):
    """
    'antmaze-giant-navigate-v0'
      → maze_type='antmaze', size='giant', task_type='navigate'
    Returns (maze_type, size, task_type) or None if pattern not recognised.
    """
    TYPE_WORDS = {"navigate", "stitch", "explore"}
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
    """'antmaze-giant-navigate-v0' → 'antmaze-giant-v0'"""
    TYPE_WORDS = {"navigate", "stitch", "explore"}
    parts = name.split("-")
    kept = [p for p in parts if p not in TYPE_WORDS]
    return "-".join(kept)

def config_stem_from_name(name):
    """'antmaze-giant-navigate-v0' → 'og_antmaze_giant_navigate'"""
    # Remove -v0 (or -v1 etc.) suffix, then replace hyphens with underscores
    import re
    cleaned = re.sub(r"-v\d+$", "", name)
    return "og_" + cleaned.replace("-", "_")

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

    # episode_len lookup
    key = (maze_type, size.replace("_", "-") if "_" in size else size, task_type)
    # normalise size key: "giant" not "giant-v0"
    episode_len = EPISODE_LEN_TABLE.get(key)
    if episode_len is None:
        print(f"[warn] {npz_path.name}: no episode_len rule for {key}, defaulting to 500")
        episode_len = 500

    print(f"Processing: {npz_path.name}")
    print(f"  → {config_path.name}  (episode_len={episode_len})")

    try:
        data = np.load(str(npz_path), allow_pickle=False)
        obs       = data["observations"]   # (N, obs_dim) — always 29D for antmaze
        terminals = data["terminals"]      # (N,) bool

        n_samples = len(obs)
        obs_dim   = obs.shape[1]

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

        # ── Format output ──────────────────────────────────────────────────
        def fmt_list(vals, precision=8):
            return "[" + ", ".join(f"{v:.{precision}g}" for v in vals) + "]"

        lines = [
            f"defaults:",
            f"  - base_dataset",
            f"",
            f"# ── [Group A: Dataset Loader] ─────────────────────────────────────────────",
            f"# Read directly by OGAntMazeOfflineRLDataset.__init__() and env creation.",
            f"# Required at both train and eval time.",
            f'env_id: "{env_id}"',
            f'dataset: "{stem}"',
            f"save_dir: ~/.ogbench/data",
            f"episode_len: {episode_len}",
            f"num_tasks: {NUM_TASKS}",
            f"",
            f"# ── [Group B: Normalization Stats] ────────────────────────────────────────",
            f"# train time: interpolated into algorithm cfg via train_df_planning.yaml",
            f"#             (${'{'}dataset.observation_mean{'}'} etc.) and saved to ckpt training_hparams.",
            f"# eval time:  restored from ckpt training_hparams; yaml value is NOT used.",
            f"# Computed from {stem} training data",
            f"# ({n_samples} samples, obs_dim={obs_dim} = qpos[15] + qvel[14]).",
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

        print(f"  obs_dim={obs_dim}, n_samples={n_samples}, "
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
