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
DOCKER_IMAGE="mctd:0.1"
DOCKER_USER="jmseo1204"
DOCKER_PROJECT="/home/$DOCKER_USER/mctd"
DOCKER_OUTPUTS="/home/$DOCKER_USER/mctd/outputs"
OUTPUT_MOUNT_DIR="/home/jmseo1204/mctd_outputs"
OGBENCH_DATA_DIR="/mnt/c/Users/USER/Desktop/test_ogbench/ogbench_data"
HOME_DIR="$HOME"

EVAL_BASE="$PROJECT_DIR/outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_MOUNT_DIR"

# ────────────────────────────────────────────────────────
# Step 1: Select state dimension
# ────────────────────────────────────────────────────────
echo "===================================================="
echo "  MCTD Training Launcher"
echo "===================================================="
echo "Select observation state dimension:"
echo "  1) 2D  (x,y position only)   [og_antmaze_giant_navigate]"
echo "  2) 15D (qpos only)            [og_antmaze_giant_navigate_15d]"
echo "  3) 29D (full qpos+qvel)       [og_antmaze_giant_navigate_fullstate]"
echo ""
read -p "Enter [1-3]: " DIM_SEL

case "$DIM_SEL" in
    1)
        TARGET_OBS_DIM=2
        ALGORITHM_CONFIG="df_planning_2d"
        DATASET_CONFIG="og_antmaze_giant_navigate"
        DEFAULT_JUMP=5
        RUN_NAME="Train_2D_antmaze"
        MODEL_ID_PREFIX="train_2d"
        LOG_FILE="$PROJECT_DIR/logs/train_2d.log"
        CKPT_MAP_FILE="$PROJECT_DIR/logs/2d_ckpt_model_map.txt"
        MODEL_ID_FILE="$PROJECT_DIR/logs/current_2d_model_id.txt"
        ;;
    2)
        TARGET_OBS_DIM=15
        ALGORITHM_CONFIG="df_planning_15d"
        DATASET_CONFIG="og_antmaze_giant_navigate_15d"
        DEFAULT_JUMP=1
        RUN_NAME="Train_15D_antmaze"
        MODEL_ID_PREFIX="train_15d"
        LOG_FILE="$PROJECT_DIR/logs/train_15d.log"
        CKPT_MAP_FILE="$PROJECT_DIR/logs/15d_ckpt_model_map.txt"
        MODEL_ID_FILE="$PROJECT_DIR/logs/current_15d_model_id.txt"
        ;;
    3)
        TARGET_OBS_DIM=29
        ALGORITHM_CONFIG="df_planning"
        DATASET_CONFIG="og_antmaze_giant_navigate_fullstate"
        DEFAULT_JUMP=1
        RUN_NAME="Train_29D_big"
        MODEL_ID_PREFIX="train_29d"
        LOG_FILE="$PROJECT_DIR/logs/train_29d.log"
        CKPT_MAP_FILE="$PROJECT_DIR/logs/29d_ckpt_model_map.txt"
        MODEL_ID_FILE="$PROJECT_DIR/logs/current_29d_model_id.txt"
        ;;
    *)
        echo "Invalid selection '$DIM_SEL'. Exiting."
        exit 1
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
# update_eval_symlink: find latest last.ckpt in OUTPUT_MOUNT_DIR
#   and symlink it into EVAL_BASE/$model_id/model.ckpt
# ────────────────────────────────────────────────────────
update_eval_symlink() {
    local model_id="$1"
    local eval_dir="$EVAL_BASE/$model_id"
    mkdir -p "$eval_dir"

    local latest_ckpt=""
    local latest_time=0
    while IFS= read -r f; do
        local ftime
        ftime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        if [ "$ftime" -gt "$latest_time" ]; then
            latest_time=$ftime
            latest_ckpt=$f
        fi
    done < <(find "$OUTPUT_MOUNT_DIR" -name "last.ckpt" 2>/dev/null)

    if [ -z "$latest_ckpt" ]; then
        echo "[symlink] No last.ckpt found in $OUTPUT_MOUNT_DIR, skipping." | tee -a "$LOG_FILE"
        return
    fi

    local real_ckpt
    real_ckpt=$(realpath "$latest_ckpt")
    ln -sf "$real_ckpt" "$eval_dir/model.ckpt"
    echo "[symlink] $model_id/model.ckpt -> $real_ckpt" | tee -a "$LOG_FILE"

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
# Scan checkpoints via Docker (directory scan, no stdin pipe)
# ────────────────────────────────────────────────────────
echo "========================================"
echo "[$(date)] Searching for ${TARGET_OBS_DIM}D checkpoints (obs_dim=$TARGET_OBS_DIM)..."

CKPT_DIRS=()

SCANNER_SCRIPT=$(mktemp /tmp/mctd_scan_XXXXXX.py)
chmod 644 "$SCANNER_SCRIPT"
cat > "$SCANNER_SCRIPT" <<'PYEOF'
import sys, torch
from pathlib import Path

docker_outputs   = sys.argv[1]  # /home/jmseo1204/mctd/outputs  (inside Docker)
output_mount_dir = sys.argv[2]  # /home/jmseo1204/mctd_outputs   (on host)
target_dim       = int(sys.argv[3])

base = Path(docker_outputs)
if not base.exists():
    print(f"ERROR: {docker_outputs} not found inside container", file=sys.stderr)
    sys.exit(0)

for ckpt_path in sorted(base.rglob("last.ckpt")):
    # Skip downloaded/ — those are symlinks, not real checkpoints
    if "downloaded" in ckpt_path.parts:
        continue
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", {})
        dm   = sd.get("data_mean")
        if dm is None or int(dm.shape[0]) != target_dim:
            continue
        epoch = ckpt.get("epoch", 0)
        mtime = int(ckpt_path.stat().st_mtime)
        # Convert docker path → host path for shell-side use
        host_path = output_mount_dir + str(ckpt_path)[len(docker_outputs):]
        # Derive human-readable name from date/time path components
        parts = ckpt_path.parts
        try:
            idx = parts.index("outputs") + 1
            name = "/".join(parts[idx:idx+2])
        except (ValueError, IndexError):
            name = str(ckpt_path.parent)
        print(f"{name}|{host_path}|{mtime}|{epoch}")
    except Exception as e:
        print(f"SCAN_ERR {ckpt_path}: {e}", file=sys.stderr)
PYEOF

echo "[$(date)] Scanning checkpoints via Docker (filtering obs_dim=$TARGET_OBS_DIM)..."
mapfile -t FILTERED < <(
    docker run --rm \
        --entrypoint python3 \
        -v "$OUTPUT_MOUNT_DIR":"$DOCKER_OUTPUTS" \
        -v "$SCANNER_SCRIPT":"$SCANNER_SCRIPT":ro \
        "$DOCKER_IMAGE" \
        "$SCANNER_SCRIPT" "$DOCKER_OUTPUTS" "$OUTPUT_MOUNT_DIR" "$TARGET_OBS_DIM" \
        2>/dev/null | grep -E '^[^|]+\|[^|]+\|[0-9]+\|[0-9]+$'
)
rm -f "$SCANNER_SCRIPT"

for entry in "${FILTERED[@]:-}"; do
    [ -n "$entry" ] && CKPT_DIRS+=("$entry")
done

# ────────────────────────────────────────────────────────
# Present checkpoint menu and let user select
# ────────────────────────────────────────────────────────
SELECTED_CKPT=""
SELECTED_EPOCH=0
MODEL_ID=""

if [ ${#CKPT_DIRS[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Found ${#CKPT_DIRS[@]} checkpoint(s) with obs_dim=${TARGET_OBS_DIM}:"
    echo "========================================"

    mapfile -t sorted < <(printf '%s\n' "${CKPT_DIRS[@]}" | sort -t'|' -k3 -rn)

    echo "  [0] Start from scratch (fresh training)"
    for i in "${!sorted[@]}"; do
        entry="${sorted[$i]}"
        ckpt_name=$(echo "$entry" | cut -d'|' -f1)
        ckpt_path=$(echo "$entry" | cut -d'|' -f2)
        epoch_num=$(echo "$entry" | cut -d'|' -f4)
        epoch_num="${epoch_num:-0}"
        [ -z "$ckpt_path" ] && continue
        ckpt_dir=$(dirname "$(realpath "$ckpt_path" 2>/dev/null || echo "$ckpt_path")")
        mapped_id=""
        if [ -f "$CKPT_MAP_FILE" ]; then
            mapped_id=$(grep "^$ckpt_dir|" "$CKPT_MAP_FILE" | cut -d'|' -f2 | tail -1 || true)
        fi
        id_hint=""
        [ -n "$mapped_id" ] && id_hint=" [eval: $mapped_id]"
        printf "  [%d] %s  (epoch %s)%s\n" "$((i+1))" "$ckpt_name" "$epoch_num" "$id_hint"
    done
    echo ""

    read -p "Select checkpoint to resume [0-${#sorted[@]}]: " SELECTION

    if [ "$SELECTION" = "0" ] || [ -z "$SELECTION" ]; then
        SELECTED_CKPT=""
        SELECTED_EPOCH=0
        echo "Starting fresh training."
    else
        idx=$((SELECTION - 1))
        if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#sorted[@]}" ]; then
            entry="${sorted[$idx]}"
            SELECTED_CKPT=$(echo "$entry" | cut -d'|' -f2)
            SELECTED_EPOCH=$(echo "$entry" | cut -d'|' -f4)
            SELECTED_EPOCH="${SELECTED_EPOCH:-0}"
            ckpt_name_sel=$(echo "$entry" | cut -d'|' -f1)
            echo "Resuming from: $SELECTED_CKPT  (epoch $SELECTED_EPOCH)"

            ckpt_dir=$(dirname "$(realpath "$SELECTED_CKPT" 2>/dev/null || echo "$SELECTED_CKPT")")
            if [ -f "$CKPT_MAP_FILE" ]; then
                MODEL_ID=$(grep "^$ckpt_dir|" "$CKPT_MAP_FILE" | cut -d'|' -f2 | tail -1 || true)
            fi
            if [ -z "$MODEL_ID" ] && [[ "$ckpt_name_sel" == train_* ]]; then
                MODEL_ID="$ckpt_name_sel"
                echo "[model_id] Using checkpoint name as model_id: $MODEL_ID"
            fi
        else
            echo "Invalid selection. Starting fresh training."
            SELECTED_CKPT=""
            SELECTED_EPOCH=0
        fi
    fi
else
    echo "No existing ${TARGET_OBS_DIM}D checkpoints found. Starting fresh training."
fi

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
        done < <(find "$OUTPUT_MOUNT_DIR" -name "last.ckpt" 2>/dev/null)
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
