#!/usr/bin/env bash
# train.sh
# Unified training script for 2D / 15D / 29D AntMaze models
# - Prompts user to select state dimension at startup
# - Filters checkpoints by obs_dim via Docker Python scan
# - Checkpoint named: train_{dim}d_{epoch}ep_{YYYYMMDDHHMMSS}[_{postfix}]
# - Uses Docker (mctd:0.1) — no conda dependency

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load central user config (DOCKER_USER) before anything else
# shellcheck source=scripts/project_config.sh
source "$PROJECT_DIR/scripts/project_config.sh"

# Docker configuration (matches run_jobs.py / train_interactive.sh)
DOCKER_PROJECT="/home/$DOCKER_USER/mctd"
OGBENCH_DATA_DIR="$(dirname "$PROJECT_DIR")/ogbench_data"
HOME_DIR="$HOME"

# Source shared checkpoint utilities (sets MCTD_DOCKER_IMAGE, MCTD_DOCKER_OUTPUTS,
# MCTD_OUTPUT_MOUNT_DIR, MCTD_EVAL_BASE, and all mctd_* functions)
MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

# Aliases for backward-compat with the rest of this script
DOCKER_IMAGE="$MCTD_DOCKER_IMAGE"
DOCKER_OUTPUTS="$MCTD_DOCKER_OUTPUTS"
OUTPUT_MOUNT_DIR="$MCTD_OUTPUT_MOUNT_DIR"
EVAL_BASE="$MCTD_EVAL_BASE"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_MOUNT_DIR"

# ────────────────────────────────────────────────────────
# GPU selection (before interactive menus)
# ────────────────────────────────────────────────────────
mctd_select_gpus

# ────────────────────────────────────────────────────────
# Read training parameters from train_df_planning.yaml
# (no user prompts for dataset / jump — edit the YAML to change them)
# ────────────────────────────────────────────────────────
echo "===================================================="
echo "  MCTD Training Launcher"
echo "===================================================="

ALGORITHM_CONFIG="train_df_planning"
TRAIN_YAML="$PROJECT_DIR/configurations/algorithm/train_df_planning.yaml"
DF_YAML="$PROJECT_DIR/configurations/algorithm/df_planning.yaml"
DF_BASE_YAML="$PROJECT_DIR/configurations/algorithm/df_base.yaml"

# Read training parameters from YAML (inline Python, yaml is stdlib-available)
eval "$(python3 - "$TRAIN_YAML" "$DF_YAML" "$DF_BASE_YAML" <<'PYEOF'
import sys, yaml, json, shlex, os

def load_yaml(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in ('defaults',):
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

train_path, df_path, df_base_path = sys.argv[1], sys.argv[2], sys.argv[3]
merged = deep_merge(deep_merge(load_yaml(df_base_path), load_yaml(df_path)), load_yaml(train_path))

dataset_config = str(merged.get('train_dataset_config', 'og_antmaze_giant_stitch'))
jump           = str(merged.get('train_jump', 5))
obs_idx        = merged.get('obs_dim_indices')
pos_idx        = merged.get('pos_dim_indices', [0, 1])
obs_dim        = str(len(obs_idx)) if obs_idx else 'unknown'
obs_dim_json   = json.dumps(obs_idx) if obs_idx else 'null'
pos_dim_json   = json.dumps(list(pos_idx)) if pos_idx else '[0,1]'
frame_stack    = str(merged.get('frame_stack', 'unknown'))
arch = (merged.get('diffusion') or {}).get('architecture', {})
network_size = str(arch.get('network_size', 'unknown'))
num_layers   = str(arch.get('num_layers', 'unknown'))
attn_heads   = str(arch.get('attn_heads', 'unknown'))

print(f"DATASET_CONFIG={shlex.quote(dataset_config)}")
print(f"JUMP_VALUE={shlex.quote(jump)}")
print(f"TARGET_OBS_DIM={shlex.quote(obs_dim)}")
print(f"OBS_DIM_JSON={shlex.quote(obs_dim_json)}")
print(f"POS_DIM_JSON={shlex.quote(pos_dim_json)}")
print(f"TRAIN_FRAME_STACK={shlex.quote(frame_stack)}")
print(f"TRAIN_NETWORK_SIZE={shlex.quote(network_size)}")
print(f"TRAIN_NUM_LAYERS={shlex.quote(num_layers)}")
print(f"TRAIN_ATTN_HEADS={shlex.quote(attn_heads)}")
PYEOF
)"

echo "Read training config from $TRAIN_YAML:"
echo "  dataset_config  = $DATASET_CONFIG"
echo "  jump            = $JUMP_VALUE"
echo "  obs_dim         = $TARGET_OBS_DIM  (obs_dim_indices=$OBS_DIM_JSON)"
echo "  frame_stack     = $TRAIN_FRAME_STACK"
echo "  network         = size=$TRAIN_NETWORK_SIZE  layers=$TRAIN_NUM_LAYERS  heads=$TRAIN_ATTN_HEADS"
echo ""

# Derived display values
DATASET_KEYWORDS=$(echo "$DATASET_CONFIG" \
    | sed 's/^og_//' \
    | sed 's/_\(2d\|15d\|29d\|fullstate\)$//')
MODEL_ID_PREFIX="train_${TARGET_OBS_DIM}d"
LOG_FILE="$PROJECT_DIR/logs/train_${TARGET_OBS_DIM}d.log"
MODEL_ID_FILE="$PROJECT_DIR/logs/current_${TARGET_OBS_DIM}d_model_id.txt"
RUN_NAME="Train_${TARGET_OBS_DIM}D_antmaze"

# ────────────────────────────────────────────────────────
# Trap: log any unexpected exit/error/signal
# ────────────────────────────────────────────────────────
_trap_handler() {
    local exit_code=$?
    local signal="${1:-EXIT}"
    if [ "$signal" = "EXIT" ] && [ "$exit_code" -eq 0 ]; then
        return
    fi
    echo "[$(date)] [FATAL] train.sh terminated. signal=$signal exit_code=$exit_code line=$BASH_LINENO" | tee -a "$LOG_FILE"
    local frame=0
    while caller $frame >> "$LOG_FILE" 2>/dev/null; do
        frame=$((frame + 1))
    done
}
trap '_trap_handler EXIT'  EXIT
trap '_trap_handler INT;  exit 130' INT
trap '_trap_handler TERM; exit 143' TERM
trap '_trap_handler ERR'  ERR

# ────────────────────────────────────────────────────────
# host_to_docker_path: convert host absolute path → Docker-internal path
# ────────────────────────────────────────────────────────
host_to_docker_path() {
    local p="$1"
    if [[ "$p" == "$OUTPUT_MOUNT_DIR"* ]]; then
        echo "${DOCKER_OUTPUTS}${p#$OUTPUT_MOUNT_DIR}"
    elif [[ "$p" == "$PROJECT_DIR"* ]]; then
        echo "${DOCKER_PROJECT}${p#$PROJECT_DIR}"
    else
        echo "$p"
    fi
}

# ────────────────────────────────────────────────────────
# update_eval_symlink: find latest model.ckpt (training output) in OUTPUT_MOUNT_DIR
#   and symlink it into EVAL_BASE/$model_id/model.ckpt using a relative path.
# ────────────────────────────────────────────────────────
update_eval_symlink() {
    local model_id="$1"

    # Find latest model.ckpt in raw training outputs only (outputs/ subdir)
    local latest_ckpt="" latest_time=0
    while IFS= read -r f; do
        local ftime
        ftime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        if [ "$ftime" -gt "$latest_time" ]; then
            latest_time=$ftime
            latest_ckpt=$f
        fi
    done < <(find "$OUTPUT_MOUNT_DIR" \
        -regextype posix-extended \
        -regex ".*/[0-9]{4}-[0-9]{2}-[0-9]{2}/.*" \
        -name "model.ckpt" 2>/dev/null)

    if [ -z "$latest_ckpt" ]; then
        echo "[symlink] No model.ckpt found in $OUTPUT_MOUNT_DIR, skipping." | tee -a "$LOG_FILE"
        return
    fi

    local real_ckpt
    real_ckpt=$(realpath "$latest_ckpt")

    # Read actual epoch from checkpoint via Docker
    local docker_ckpt_path
    docker_ckpt_path=$(host_to_docker_path "$real_ckpt")
    local actual_epoch
    actual_epoch=$(docker run --rm --entrypoint python3 \
        -v "$OUTPUT_MOUNT_DIR":"$DOCKER_OUTPUTS" \
        "$DOCKER_IMAGE" \
        -c "import torch,sys; ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(ck.get('epoch', 0))" \
        "$docker_ckpt_path" 2>/dev/null | grep -E '^[0-9]+$' | head -1 || true)
    actual_epoch="${actual_epoch:-0}"

    # Update epoch in model_id: replace _Xep_ with _${actual_epoch}ep_
    local new_model_id
    new_model_id=$(echo "$model_id" | sed -E "s/_[0-9]+ep_/_${actual_epoch}ep_/")

    # If model_id changed, rename eval dir and update globals
    if [ "$new_model_id" != "$model_id" ]; then
        local old_eval_dir="$EVAL_BASE/$model_id"
        local new_eval_dir="$EVAL_BASE/$new_model_id"
        if [ -d "$old_eval_dir" ]; then
            mv "$old_eval_dir" "$new_eval_dir" 2>/dev/null || mkdir -p "$new_eval_dir"
        else
            mkdir -p "$new_eval_dir"
        fi
        echo "[symlink] Renamed model_id: $model_id → $new_model_id  (epoch=$actual_epoch)" | tee -a "$LOG_FILE"
        MODEL_ID="$new_model_id"
        echo "$MODEL_ID" > "$MODEL_ID_FILE"
        model_id="$new_model_id"
    else
        mkdir -p "$EVAL_BASE/$model_id"
    fi

    local eval_dir="$EVAL_BASE/$model_id"
    # Use a relative symlink so it resolves correctly both on host and inside Docker
    local rel_path
    rel_path=$(realpath --relative-to="$eval_dir" "$real_ckpt")
    ln -sf "$rel_path" "$eval_dir/model.ckpt"
    echo "[symlink] $model_id/model.ckpt -> $rel_path  (epoch=$actual_epoch)" | tee -a "$LOG_FILE"
}

# ────────────────────────────────────────────────────────
# Kill existing training processes
# ────────────────────────────────────────────────────────
echo "========================================"
echo "[$(date)] Checking for existing training processes..."

EXISTING_PIDS=$(ps aux | grep -E "docker.*mctd_training|python.*main.py.*$RUN_NAME" | grep -v grep | awk '{print $2}' | grep -v $$ || true)

if [ -n "$EXISTING_PIDS" ]; then
    echo "Found existing training processes: $EXISTING_PIDS"
    echo "Killing existing processes..."
    echo "$EXISTING_PIDS" | xargs -r kill -9 2>/dev/null || true
    docker rm -f mctd_training 2>/dev/null || true
    sleep 2
    echo "Existing processes killed."
else
    echo "No existing training processes found."
    docker rm -f mctd_training 2>/dev/null || true
fi

# ────────────────────────────────────────────────────────
# GPU availability check (after container cleanup to avoid false positives)
# ────────────────────────────────────────────────────────
mctd_check_gpu_availability

# ────────────────────────────────────────────────────────
# Scan checkpoints via Docker and filter by arch match
# ────────────────────────────────────────────────────────
echo "========================================"
echo "[$(date)] Scanning all checkpoints via Docker..."

mctd_scan_ckpts 0   # 0 = no obs_dim filter; show all

# Filter: keep only checkpoints matching the current train_df_planning.yaml arch.
# A field value of "unknown" in the checkpoint is treated as a wildcard (matches anything).
MCTD_CKPT_DIRS_FILTERED=()
for _entry in "${MCTD_CKPT_DIRS[@]:-}"; do
    [ -z "$_entry" ] && continue
    _c_obs=$(   echo "$_entry" | cut -d'|' -f6)
    _c_jump=$(  echo "$_entry" | cut -d'|' -f7)
    _c_ds=$(    echo "$_entry" | cut -d'|' -f8)
    _c_fs=$(    echo "$_entry" | cut -d'|' -f9)
    _c_net=$(   echo "$_entry" | cut -d'|' -f10)
    _c_layers=$(echo "$_entry" | cut -d'|' -f11)
    _c_heads=$( echo "$_entry" | cut -d'|' -f12)
    # For each field: skip only if both are known AND they differ
    _match=1
    [[ "$_c_obs"    != "unknown" && "$_c_obs"    != "$TARGET_OBS_DIM"     ]] && _match=0
    [[ "$_c_jump"   != "unknown" && "$_c_jump"   != "$JUMP_VALUE"         ]] && _match=0
    [[ "$_c_ds"     != "unknown" && "$_c_ds"     != "$DATASET_CONFIG"     ]] && _match=0
    [[ "$_c_fs"     != "unknown" && "$_c_fs"     != "$TRAIN_FRAME_STACK"  ]] && _match=0
    [[ "$_c_net"    != "unknown" && "$_c_net"    != "$TRAIN_NETWORK_SIZE" ]] && _match=0
    [[ "$_c_layers" != "unknown" && "$_c_layers" != "$TRAIN_NUM_LAYERS"   ]] && _match=0
    [[ "$_c_heads"  != "unknown" && "$_c_heads"  != "$TRAIN_ATTN_HEADS"   ]] && _match=0
    [ "$_match" -eq 1 ] && MCTD_CKPT_DIRS_FILTERED+=("$_entry")
done
MCTD_CKPT_DIRS=("${MCTD_CKPT_DIRS_FILTERED[@]:-}")
echo "[$(date)] Found ${#MCTD_CKPT_DIRS[@]} matching checkpoint(s) for current arch config."

CKPT_DIRS=("${MCTD_CKPT_DIRS[@]:-}")

# ────────────────────────────────────────────────────────
# Present checkpoint menu and let user select
# ────────────────────────────────────────────────────────
mctd_ckpt_menu

SELECTED_CKPT="${MCTD_SELECTED_CKPT:-}"
SELECTED_EPOCH="${MCTD_SELECTED_EPOCH:-0}"
MODEL_ID="${MCTD_SELECTED_MODEL_ID:-}"

# ────────────────────────────────────────────────────────
# Ask user for optional name postfix
# ────────────────────────────────────────────────────────
echo ""
read -p "Optional name postfix (leave blank for none): " USER_POSTFIX
USER_POSTFIX="${USER_POSTFIX// /_}"

# ────────────────────────────────────────────────────────
# Build MODEL_ID: train_{dim}d_{dataset}_{epoch}ep_{YYYYMMDD}_{HHMMSS}[_{postfix}]
# ────────────────────────────────────────────────────────
if [ -z "$MODEL_ID" ]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    MODEL_ID="${MODEL_ID_PREFIX}_${DATASET_KEYWORDS}_j${JUMP_VALUE}_${SELECTED_EPOCH}ep_${TIMESTAMP}"
    if [ -n "$USER_POSTFIX" ]; then
        MODEL_ID="${MODEL_ID}_${USER_POSTFIX}"
    fi
    echo "[model_id] New model_id: $MODEL_ID"
fi
echo "$MODEL_ID" > "$MODEL_ID_FILE"
echo "[model_id] Using: $MODEL_ID" | tee -a "$LOG_FILE"

# ────────────────────────────────────────────────────────
# save_training_config: persist training-dependent architecture params
#   so that generate_jobs_generalized.py can auto-detect them at eval time.
#   Reads configurations/algorithm/{algo}.yaml (+ parent df_planning.yaml) and
#   writes EVAL_BASE/$model_id/training_config.yaml in plain-YAML format that
#   extract_from_config() already handles (no WandB value-wrapper needed).
# ────────────────────────────────────────────────────────
save_training_config() {
    local model_id="$1"
    local algo_config="$2"
    local dataset_config="$3"   # e.g. og_antmaze_giant_stitch
    local jump_value="$4"       # e.g. 5
    local obs_dim_json="$5"     # JSON list e.g. [0,1]
    local pos_dim_json="$6"     # JSON list e.g. [0,1]

    local base_yaml="$PROJECT_DIR/configurations/algorithm/df_planning.yaml"
    local algo_yaml="$PROJECT_DIR/configurations/algorithm/${algo_config}.yaml"
    local dataset_yaml="$PROJECT_DIR/configurations/dataset/${dataset_config}.yaml"
    local output_dir="$EVAL_BASE/$model_id"
    local output_file="$output_dir/training_config.yaml"

    mkdir -p "$output_dir"

    python3 - "$base_yaml" "$algo_yaml" "$dataset_yaml" "$dataset_config" "$jump_value" \
              "$obs_dim_json" "$pos_dim_json" "$output_file" <<'PYEOF'
import sys, yaml, json

def load_yaml(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k == 'defaults':
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

base_path, algo_path, dataset_path, dataset_config_name, jump_value, \
    obs_dim_json, pos_dim_json, out_path = sys.argv[1:]

merged_algo  = deep_merge(load_yaml(base_path), load_yaml(algo_path))
dataset_data = load_yaml(dataset_path)

arch = (merged_algo.get('diffusion') or {}).get('architecture', {})

# Resolve episode_len: algo yaml may use ${dataset.episode_len} interpolation
episode_len_raw = merged_algo.get('episode_len')
if isinstance(episode_len_raw, str) and '${' in episode_len_raw:
    episode_len = dataset_data.get('episode_len')
else:
    episode_len = episode_len_raw or dataset_data.get('episode_len')

obs_dim_indices = json.loads(obs_dim_json) if obs_dim_json != 'null' else None
pos_dim_indices = json.loads(pos_dim_json) if pos_dim_json not in ('null', '') else [0, 1]

config_to_save = {
    'algorithm': {
        'causal': merged_algo.get('causal'),
        'scheduling_matrix': merged_algo.get('scheduling_matrix'),
        'frame_stack': merged_algo.get('frame_stack'),
        'obs_dim_indices': obs_dim_indices,
        'pos_dim_indices': pos_dim_indices,
        'diffusion': {
            'architecture': {
                'attn_heads': arch.get('attn_heads'),
                'network_size': arch.get('network_size'),
                'dim_feedforward': arch.get('dim_feedforward'),
                'num_layers': arch.get('num_layers'),
            }
        }
    },
    'dataset': {
        'config': dataset_config_name,
        'episode_len': episode_len,
        'jump': int(jump_value) if jump_value.isdigit() else dataset_data.get('jump', 1),
    }
}

with open(out_path, 'w') as f:
    yaml.dump(config_to_save, f, default_flow_style=False)
print(f"[config] Saved training config to {out_path}")
PYEOF

    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[config] WARNING: Failed to save training_config.yaml (exit $rc)" | tee -a "$LOG_FILE"
    fi
}

# Save training-dependent config for later eval job generation
save_training_config "$MODEL_ID" "$ALGORITHM_CONFIG" "$DATASET_CONFIG" "$JUMP_VALUE" \
    "$OBS_DIM_JSON" "$POS_DIM_JSON"

# ────────────────────────────────────────────────────────
# Training loop (Docker-based)
# ────────────────────────────────────────────────────────
echo "========================================" | tee -a "$LOG_FILE"
echo "[$(date)] Starting ${TARGET_OBS_DIM}D AntMaze training" | tee -a "$LOG_FILE"
echo "Project dir : $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Output dir  : $OUTPUT_MOUNT_DIR" | tee -a "$LOG_FILE"
echo "Docker image: $DOCKER_IMAGE" | tee -a "$LOG_FILE"
echo "Algorithm   : $ALGORITHM_CONFIG" | tee -a "$LOG_FILE"
echo "Dataset     : $DATASET_CONFIG  (jump=$JUMP_VALUE)" | tee -a "$LOG_FILE"
echo "Model ID    : $MODEL_ID" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

LOAD_CKPT=""
if [ -n "$SELECTED_CKPT" ]; then
    LOAD_CKPT="$(host_to_docker_path "$SELECTED_CKPT")"
fi

BASE_CMD="python3 main.py \
    experiment.tasks=[training] \
    experiment=exp_planning \
    algorithm=$ALGORITHM_CONFIG \
    dataset=$DATASET_CONFIG \
    dataset.jump=$JUMP_VALUE \
    +name=$MODEL_ID \
    experiment.training.data.num_workers=0 \
    experiment.validation.data.num_workers=0 \
    experiment.validation.limit_batch=0"

if [ -n "$LOAD_CKPT" ]; then
    echo "[$(date)] Resuming from (docker path): $LOAD_CKPT" | tee -a "$LOG_FILE"
    INNER_CMD="$BASE_CMD +load=$LOAD_CKPT"
else
    echo "[$(date)] Starting fresh training" | tee -a "$LOG_FILE"
    INNER_CMD="$BASE_CMD"
fi

# Derive CUDA_VISIBLE_DEVICES from AVAILABLE_GPUS (e.g. "localhost:4,localhost:5" → "4,5")
_gpu_ids=$(echo "$AVAILABLE_GPUS" | tr ',' '\n' | grep '^localhost:' | sed 's/localhost://' | tr '\n' ',' | sed 's/,$//')
_CUDA_VIS_FLAG=""
[ -n "$_gpu_ids" ] && _CUDA_VIS_FLAG="-e CUDA_VISIBLE_DEVICES=${_gpu_ids}"

FULL_CMD="docker run --rm --gpus all ${_CUDA_VIS_FLAG} --name mctd_training --shm-size=8g \
    -e MUJOCO_GL=osmesa \
    -e HYDRA_FULL_ERROR=1 \
    -e WANDB_ENTITY=$WANDB_ENTITY \
    -e WANDB_PROJECT=$WANDB_PROJECT \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/$DOCKER_USER/.mujoco/mujoco210/bin \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -v $PROJECT_DIR:$DOCKER_PROJECT \
    -v $OUTPUT_MOUNT_DIR:$DOCKER_OUTPUTS \
    -v $OGBENCH_DATA_DIR:/home/$DOCKER_USER/.ogbench/data \
    -v $HOME_DIR/.netrc:/home/$DOCKER_USER/.netrc \
    -v $HOME_DIR/.d4rl:/home/$DOCKER_USER/.d4rl \
    $DOCKER_IMAGE /bin/bash \
    -c 'cd $DOCKER_PROJECT && git config --global --add safe.directory $DOCKER_PROJECT 2>/dev/null; $INNER_CMD'"

echo "[$(date)] Docker command: $FULL_CMD" | tee -a "$LOG_FILE"

set +e
eval "$FULL_CMD" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

update_eval_symlink "$MODEL_ID"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Training completed successfully!" | tee -a "$LOG_FILE"
    exit 0
fi

echo "[$(date)] Training failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
exit 1
