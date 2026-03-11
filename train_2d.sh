#!/usr/bin/env bash
# train_2d.sh
# Trains the 2D AntMaze model (obs_dim=2: x,y position only)
# - wandb.mode=offline: never blocks on network
# - Auto-restarts on crash, up to MAX_RETRIES
# - Filters checkpoints by obs_dim=2 via Docker python scan
# - Checkpoint named: train_2d_{epoch}ep_{YYYYMMDDHHMMSS}[_{postfix}]
# - Uses Docker (mctd:0.1) — no conda dependency

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/logs/train_2d.log"
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

EVAL_BASE="$OUTPUT_MOUNT_DIR/downloaded/jmseo1204-seoul-national-university/mctd_eval"
MODEL_ID_FILE="$PROJECT_DIR/logs/current_2d_model_id.txt"
CKPT_MAP_FILE="$PROJECT_DIR/logs/2d_ckpt_model_map.txt"

TARGET_OBS_DIM=2

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_MOUNT_DIR"

# ────────────────────────────────────────────────────────
# Trap: log any unexpected exit/error/signal
# ────────────────────────────────────────────────────────
_trap_handler() {
    local exit_code=$?
    local signal="${1:-EXIT}"
    if [ "$signal" = "EXIT" ] && [ "$exit_code" -eq 0 ]; then
        return
    fi
    echo "[$(date)] [FATAL] train_2d.sh terminated. signal=$signal exit_code=$exit_code line=$BASH_LINENO" | tee -a "$LOG_FILE"
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

EXISTING_PIDS=$(ps aux | grep -E "docker.*train_2d_training|python.*main.py.*Train_2D_antmaze" | grep -v grep | awk '{print $2}' | grep -v $$ || true)

if [ -n "$EXISTING_PIDS" ]; then
    echo "Found existing training processes: $EXISTING_PIDS"
    echo "Killing existing processes..."
    echo "$EXISTING_PIDS" | xargs -r kill -9 2>/dev/null || true
    docker rm -f train_2d_training 2>/dev/null || true
    sleep 2
    echo "Existing processes killed."
else
    echo "No existing training processes found."
    docker rm -f train_2d_training 2>/dev/null || true
fi

# ────────────────────────────────────────────────────────
# Collect candidate checkpoint paths
# ────────────────────────────────────────────────────────
echo "========================================"
echo "[$(date)] Searching for 2D checkpoints (obs_dim=$TARGET_OBS_DIM)..."

RAW_CKPTS=()

# Regular training outputs (non-downloaded) in OUTPUT_MOUNT_DIR
while IFS= read -r ckpt; do
    run_dir=$(dirname "$(dirname "$(dirname "$ckpt")")")
    run_name=$(basename "$run_dir")
    RAW_CKPTS+=("$run_name|$ckpt")
done < <(find "$OUTPUT_MOUNT_DIR" -name "last.ckpt" 2>/dev/null)

# Eval symlinks in EVAL_BASE (local_* and other registered checkpoints)
if [ -d "$EVAL_BASE" ]; then
    while IFS= read -r ckpt; do
        model_id=$(basename "$(dirname "$ckpt")")
        real_ckpt=$(realpath "$ckpt" 2>/dev/null || echo "$ckpt")
        # Avoid duplicates with the above (same real path)
        RAW_CKPTS+=("$model_id|$real_ckpt")
    done < <(find "$EVAL_BASE" -name "model.ckpt" 2>/dev/null)
fi

CKPT_DIRS=()

if [ ${#RAW_CKPTS[@]} -gt 0 ]; then
    echo "[$(date)] Scanning ${#RAW_CKPTS[@]} checkpoint(s) via Docker..."

    # Write scanner script to a temp file (mounted into Docker at same path)
    SCANNER_SCRIPT=$(mktemp /tmp/train2d_scan_XXXXXX.py)
    cat > "$SCANNER_SCRIPT" <<'PYEOF'
import sys, torch
from pathlib import Path

output_mount_dir = sys.argv[1]
docker_outputs   = sys.argv[2]
project_dir      = sys.argv[3]
docker_project   = sys.argv[4]
target_dim       = int(sys.argv[5])

def host_to_docker(p):
    if p.startswith(output_mount_dir):
        return docker_outputs + p[len(output_mount_dir):]
    if p.startswith(project_dir):
        return docker_project + p[len(project_dir):]
    return p

seen = set()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    name, host_path = line.split("|", 1)
    docker_path = host_to_docker(host_path)
    if docker_path in seen:
        continue
    seen.add(docker_path)
    p = Path(docker_path)
    if not p.exists():
        continue
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", {})
        dm   = sd.get("data_mean")
        if dm is None or int(dm.shape[0]) != target_dim:
            continue
        epoch = ckpt.get("epoch", 0)
        mtime = int(p.stat().st_mtime)
        print(f"{name}|{host_path}|{mtime}|{epoch}")
    except Exception:
        pass
PYEOF

    mapfile -t FILTERED < <(
        printf '%s\n' "${RAW_CKPTS[@]}" | docker run --rm -i \
            -v "$PROJECT_DIR":"$DOCKER_PROJECT" \
            -v "$OUTPUT_MOUNT_DIR":"$DOCKER_OUTPUTS" \
            -v "$SCANNER_SCRIPT":"$SCANNER_SCRIPT":ro \
            "$DOCKER_IMAGE" \
            python3 "$SCANNER_SCRIPT" \
                "$OUTPUT_MOUNT_DIR" "$DOCKER_OUTPUTS" \
                "$PROJECT_DIR" "$DOCKER_PROJECT" \
                "$TARGET_OBS_DIM" \
            2>/dev/null
    )
    rm -f "$SCANNER_SCRIPT"

    for entry in "${FILTERED[@]:-}"; do
        [ -n "$entry" ] && CKPT_DIRS+=("$entry")
    done
fi

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

    # Sort by mtime descending (3rd field)
    mapfile -t sorted < <(printf '%s\n' "${CKPT_DIRS[@]}" | sort -t'|' -k3 -rn)

    echo "  [0] Start from scratch (fresh training)"
    for i in "${!sorted[@]}"; do
        IFS='|' read -ra parts <<< "${sorted[$i]}"
        ckpt_name="${parts[0]}"
        ckpt_path="${parts[1]}"
        epoch_num="${parts[3]:-0}"
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
            IFS='|' read -ra parts <<< "${sorted[$idx]}"
            SELECTED_CKPT="${parts[1]}"
            SELECTED_EPOCH="${parts[3]:-0}"
            echo "Resuming from: $SELECTED_CKPT  (epoch $SELECTED_EPOCH)"

            # Resolve model_id from map or checkpoint name
            ckpt_dir=$(dirname "$(realpath "$SELECTED_CKPT" 2>/dev/null || echo "$SELECTED_CKPT")")
            if [ -f "$CKPT_MAP_FILE" ]; then
                MODEL_ID=$(grep "^$ckpt_dir|" "$CKPT_MAP_FILE" | cut -d'|' -f2 | tail -1 || true)
            fi
            if [ -z "$MODEL_ID" ] && [[ "${parts[0]}" == train_* ]]; then
                MODEL_ID="${parts[0]}"
                echo "[model_id] Using checkpoint name as model_id: $MODEL_ID"
            fi
        else
            echo "Invalid selection. Starting fresh training."
            SELECTED_CKPT=""
            SELECTED_EPOCH=0
        fi
    fi
else
    echo "No existing 2D checkpoints found. Starting fresh training."
fi

# ────────────────────────────────────────────────────────
# Ask user for optional name postfix
# ────────────────────────────────────────────────────────
echo ""
read -p "Optional name postfix (leave blank for none): " USER_POSTFIX
USER_POSTFIX="${USER_POSTFIX// /_}"  # replace spaces with underscores

# ────────────────────────────────────────────────────────
# Build MODEL_ID: train_{dim}d_{epoch}ep_{YYYYMMDDHHMMSS}[_{postfix}]
# ────────────────────────────────────────────────────────
if [ -z "$MODEL_ID" ]; then
    TIMESTAMP="$(date +%Y%m%d%H%M%S)"
    MODEL_ID="train_${TARGET_OBS_DIM}d_${SELECTED_EPOCH}ep_${TIMESTAMP}"
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
echo "[$(date)] Starting 2D AntMaze training (obs_dim=$TARGET_OBS_DIM)" | tee -a "$LOG_FILE"
echo "Project dir : $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Output dir  : $OUTPUT_MOUNT_DIR" | tee -a "$LOG_FILE"
echo "Docker image: $DOCKER_IMAGE" | tee -a "$LOG_FILE"
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
        algorithm=df_planning_2d \
        dataset=og_antmaze_giant_navigate \
        dataset.jump=5 \
        +name=Train_2D_antmaze \
        wandb.mode=offline \
        experiment.validation.limit_batch=0"

    if [ -n "$LOAD_CKPT" ]; then
        echo "[$(date)] Resuming from (docker path): $LOAD_CKPT" | tee -a "$LOG_FILE"
        INNER_CMD="$BASE_CMD +load=$LOAD_CKPT"
    else
        echo "[$(date)] Starting fresh training" | tee -a "$LOG_FILE"
        INNER_CMD="$BASE_CMD"
    fi

    FULL_CMD="docker run --rm --gpus all --name train_2d_training --shm-size=50g \
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

    # Update eval symlink after each attempt
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
