#!/usr/bin/env bash
# train_29d_big.sh
# Trains the 29D AntMaze model (network_size=256, attn_heads=8, dim_feedforward=1024)
# - wandb.mode=offline: never blocks on network
# - Auto-restarts on crash, up to MAX_RETRIES
# - Resumes from latest checkpoint each restart
# - Per-dimension losses logged to logs/dim_loss.jsonl

set -euo pipefail

CONDA_ENV="diff_force_env"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/logs/train_29d_big.log"
MAX_RETRIES=20
RETRY_DELAY=30  # seconds between retries

mkdir -p "$PROJECT_DIR/logs"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "[$(date)] Starting 29D big model training" | tee -a "$LOG_FILE"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Conda env: $CONDA_ENV" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date)] Attempt $attempt / $MAX_RETRIES" | tee -a "$LOG_FILE"

    # Find latest checkpoint to resume from
    LATEST_CKPT=""
    LATEST_TIME=0
    while IFS= read -r f; do
        if [ -f "$f" ]; then
            ftime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            if [ "$ftime" -gt "$LATEST_TIME" ]; then
                LATEST_TIME=$ftime
                LATEST_CKPT=$f
            fi
        fi
    done < <(find "$PROJECT_DIR/outputs" -name "last.ckpt" ! -path "*/downloaded/*" 2>/dev/null)

    BASE_CMD="python main.py \
        experiment.tasks=[training] \
        experiment=exp_planning \
        algorithm=df_planning \
        dataset=og_antmaze_giant_navigate_fullstate \
        +name=Train_29D_big \
        wandb.mode=offline \
        experiment.validation.limit_batch=0"

    if [ -n "$LATEST_CKPT" ]; then
        echo "[$(date)] Resuming from: $LATEST_CKPT" | tee -a "$LOG_FILE"
        FULL_CMD="$BASE_CMD +load=$LATEST_CKPT"
    else
        echo "[$(date)] Starting fresh training" | tee -a "$LOG_FILE"
        FULL_CMD="$BASE_CMD"
    fi

    echo "[$(date)] Command: $FULL_CMD" | tee -a "$LOG_FILE"

    set +e
    $FULL_CMD 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

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
