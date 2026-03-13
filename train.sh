#!/usr/bin/env bash
# train.sh
# Unified training script for 2D / 15D / 29D AntMaze models
# - Prompts user to select state dimension at startup
# - Filters checkpoints by obs_dim via Docker Python scan
# - Checkpoint named: train_{dim}d_{epoch}ep_{YYYYMMDDHHMMSS}[_{postfix}]
# - Uses Docker (mctd:0.1) — no conda dependency
# - Auto-restarts on crash, up to MAX_RETRIES

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_RETRIES=20
RETRY_DELAY=30

# Docker configuration (matches run_jobs.py / train_interactive.sh)
DOCKER_USER="jmseo1204"
DOCKER_PROJECT="/home/$DOCKER_USER/mctd"
OGBENCH_DATA_DIR="/mnt/c/Users/USER/Desktop/test_ogbench/ogbench_data"
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
# Step 1: Select state dimension
# ────────────────────────────────────────────────────────
echo "===================================================="
echo "  MCTD Training Launcher"
echo "===================================================="

# Step 1: Select state dimension (shared menu → MCTD_TARGET_OBS_DIM, MCTD_DATASET_CONFIG)
mctd_dim_menu

TARGET_OBS_DIM="$MCTD_TARGET_OBS_DIM"
DATASET_CONFIG="$MCTD_DATASET_CONFIG"

# Training-specific per-dim settings
case "$TARGET_OBS_DIM" in
    2)
        ALGORITHM_CONFIG="df_planning_2d"
        DEFAULT_JUMP=5
        RUN_NAME="Train_2D_antmaze"
        MODEL_ID_PREFIX="train_2d"
        LOG_FILE="$PROJECT_DIR/logs/train_2d.log"
        CKPT_MAP_FILE="$PROJECT_DIR/logs/2d_ckpt_model_map.txt"
        MODEL_ID_FILE="$PROJECT_DIR/logs/current_2d_model_id.txt"
        ;;
    15)
        ALGORITHM_CONFIG="df_planning_15d"
        DEFAULT_JUMP=1
        RUN_NAME="Train_15D_antmaze"
        MODEL_ID_PREFIX="train_15d"
        LOG_FILE="$PROJECT_DIR/logs/train_15d.log"
        CKPT_MAP_FILE="$PROJECT_DIR/logs/15d_ckpt_model_map.txt"
        MODEL_ID_FILE="$PROJECT_DIR/logs/current_15d_model_id.txt"
        ;;
    29)
        ALGORITHM_CONFIG="df_planning"
        DEFAULT_JUMP=1
        RUN_NAME="Train_29D_big"
        MODEL_ID_PREFIX="train_29d"
        LOG_FILE="$PROJECT_DIR/logs/train_29d.log"
        CKPT_MAP_FILE="$PROJECT_DIR/logs/29d_ckpt_model_map.txt"
        MODEL_ID_FILE="$PROJECT_DIR/logs/current_29d_model_id.txt"
        ;;
esac

# ────────────────────────────────────────────────────────
# Step 2: Select jump value
# ────────────────────────────────────────────────────────
echo "Selected: ${TARGET_OBS_DIM}D  algorithm=$ALGORITHM_CONFIG  dataset=$DATASET_CONFIG"
echo ""
echo "dataset.jump: frame stride for training data (1 = use every frame)"
read -p "Enter jump value [default: $DEFAULT_JUMP]: " JUMP_INPUT
if [ -z "$JUMP_INPUT" ]; then
    JUMP_VALUE=$DEFAULT_JUMP
elif [[ "$JUMP_INPUT" =~ ^[0-9]+$ ]] && [ "$JUMP_INPUT" -ge 1 ]; then
    JUMP_VALUE=$JUMP_INPUT
else
    echo "Invalid jump value '$JUMP_INPUT'. Exiting."
    exit 1
fi
echo "Using jump=$JUMP_VALUE"
echo ""

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

    # Find latest model.ckpt in OUTPUT_MOUNT_DIR, excluding mctd_eval symlinks
    local latest_ckpt="" latest_time=0
    while IFS= read -r f; do
        local ftime
        ftime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        if [ "$ftime" -gt "$latest_time" ]; then
            latest_time=$ftime
            latest_ckpt=$f
        fi
    done < <(find "$OUTPUT_MOUNT_DIR" -name "model.ckpt" -not -path "*/mctd_eval/*" 2>/dev/null)

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

    # Update ckpt_dir → model_id map
    local ckpt_dir
    ckpt_dir=$(dirname "$real_ckpt")
    if [ -f "$CKPT_MAP_FILE" ]; then
        grep -v "^$ckpt_dir|" "$CKPT_MAP_FILE" > "${CKPT_MAP_FILE}.tmp" 2>/dev/null || true
        mv "${CKPT_MAP_FILE}.tmp" "$CKPT_MAP_FILE"
    fi
    echo "$ckpt_dir|$model_id" >> "$CKPT_MAP_FILE"
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
# Scan checkpoints via Docker
# ────────────────────────────────────────────────────────
echo "========================================"
echo "[$(date)] Searching for ${TARGET_OBS_DIM}D checkpoints (obs_dim=$TARGET_OBS_DIM)..."
echo "[$(date)] Scanning checkpoints via Docker (filtering obs_dim=$TARGET_OBS_DIM)..."

mctd_scan_ckpts "$TARGET_OBS_DIM"
CKPT_DIRS=("${MCTD_CKPT_DIRS[@]:-}")

# ────────────────────────────────────────────────────────
# Present checkpoint menu and let user select
# ────────────────────────────────────────────────────────
mctd_ckpt_menu "$TARGET_OBS_DIM"

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
# Build MODEL_ID: train_{dim}d_{epoch}ep_{YYYYMMDDHHMMSS}[_{postfix}]
# ────────────────────────────────────────────────────────
if [ -z "$MODEL_ID" ]; then
    TIMESTAMP="$(date +%Y%m%d%H%M%S)"
    MODEL_ID="${MODEL_ID_PREFIX}_j${JUMP_VALUE}_${SELECTED_EPOCH}ep_${TIMESTAMP}"
    if [ -n "$USER_POSTFIX" ]; then
        MODEL_ID="${MODEL_ID}_${USER_POSTFIX}"
    fi
    echo "[model_id] New model_id: $MODEL_ID"
fi
echo "$MODEL_ID" > "$MODEL_ID_FILE"
echo "[model_id] Using: $MODEL_ID" | tee -a "$LOG_FILE"

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

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date)] Attempt $attempt / $MAX_RETRIES" | tee -a "$LOG_FILE"

    # On attempt 1 use user-selected ckpt; on retry find latest in OUTPUT_MOUNT_DIR
    LOAD_CKPT=""
    if [ "$attempt" -eq 1 ] && [ -n "$SELECTED_CKPT" ]; then
        LOAD_CKPT="$(host_to_docker_path "$SELECTED_CKPT")"
    elif [ "$attempt" -gt 1 ]; then
        LATEST_HOST=""
        LATEST_TIME=0
        while IFS= read -r f; do
            ftime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            if [ "$ftime" -gt "$LATEST_TIME" ]; then
                LATEST_TIME=$ftime
                LATEST_HOST=$f
            fi
        done < <(find "$OUTPUT_MOUNT_DIR" -name "model.ckpt" -not -path "*/mctd_eval/*" 2>/dev/null)
        if [ -n "$LATEST_HOST" ]; then
            LOAD_CKPT="$(host_to_docker_path "$LATEST_HOST")"
        fi
    fi

    BASE_CMD="python3 main.py \
        experiment.tasks=[training] \
        experiment=exp_planning \
        algorithm=$ALGORITHM_CONFIG \
        dataset=$DATASET_CONFIG \
        dataset.jump=$JUMP_VALUE \
        +name=$RUN_NAME \
        wandb.mode=offline \
        experiment.validation.limit_batch=0"

    if [ -n "$LOAD_CKPT" ]; then
        echo "[$(date)] Resuming from (docker path): $LOAD_CKPT" | tee -a "$LOG_FILE"
        INNER_CMD="$BASE_CMD +load=$LOAD_CKPT"
    else
        echo "[$(date)] Starting fresh training" | tee -a "$LOG_FILE"
        INNER_CMD="$BASE_CMD"
    fi

    FULL_CMD="docker run --rm --gpus all --name mctd_training --shm-size=50g \
        -e MUJOCO_GL=osmesa \
        -e HYDRA_FULL_ERROR=1 \
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

    echo "[$(date)] Training crashed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "[$(date)] Waiting ${RETRY_DELAY}s before retry..." | tee -a "$LOG_FILE"
        sleep $RETRY_DELAY
    fi
done

echo "[$(date)] Max retries ($MAX_RETRIES) reached. Giving up." | tee -a "$LOG_FILE"
exit 1
