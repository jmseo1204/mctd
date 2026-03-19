#!/usr/bin/env bash

# ====================================================
# MCTD Evaluation Launcher
# ====================================================
# Prompts for state dim (2D/15D/29D) and checkpoint,
# then generates and runs evaluation jobs.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DATASET_DIR="$PROJECT_DIR/configurations/dataset"

# Source shared checkpoint utilities
MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

OUTPUT_DOWNLOADED_DIR="$MCTD_EVAL_BASE"

echo "===================================================="
echo "  MCTD Job Generator"
echo "===================================================="

# Check if Docker is available
echo "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker is not installed or not in PATH"
    exit 1
fi
if ! docker ps > /dev/null 2>&1; then
    echo "❌ ERROR: Docker daemon is not running"
    exit 1
fi
echo "✓ Docker is available and running"
echo ""

# 0. Clean up existing containers and job files
echo "Cleaning up existing MCTD containers..."
docker rm -f $(docker ps -a 2>/dev/null | grep exp_gpu0 | awk '{print $1}') 2>/dev/null || true
echo "✓ Container cleanup complete"

echo "Cleaning up existing job files..."
rm -f jobs/*.json
echo "✓ Job files cleanup complete"
echo ""

# ─────────────────────────────────────────────────────
# 1. Select state dimension
# ─────────────────────────────────────────────────────
echo "Scanning all checkpoints to identify available state dimensions..."
mctd_dim_menu || exit 1
STATE_DIM="$MCTD_TARGET_OBS_DIM"
SELECTED_DATASET="$MCTD_DATASET_CONFIG"
echo ""

DATASET_YAML="${CONFIG_DATASET_DIR}/${SELECTED_DATASET}.yaml"
if [ ! -f "$DATASET_YAML" ]; then
    echo "❌ ERROR: Dataset config not found: ${DATASET_YAML}"
    exit 1
fi

# ─────────────────────────────────────────────────────
# 2. Scan and select checkpoint
# ─────────────────────────────────────────────────────
mctd_scan_ckpts "$STATE_DIM"

if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
    echo "❌ ERROR: No checkpoints found with obs_dim=${STATE_DIM}"
    exit 1
fi

mctd_ckpt_menu "$STATE_DIM" "--no-fresh" || exit 1

# model_id is populated directly by scanner — no fallback needed
SELECTED_MODEL_ID="${MCTD_SELECTED_MODEL_ID:-}"
if [ -z "$SELECTED_MODEL_ID" ]; then
    echo "❌ ERROR: Could not determine model_id for selected checkpoint." >&2
    echo "   Scanner output is missing model_id field. Please check ckpt_scanner.py." >&2
    exit 1
fi
mctd_ensure_eval_symlink "$MCTD_SELECTED_CKPT" "$SELECTED_MODEL_ID"
echo "✓ Selected model: $SELECTED_MODEL_ID"

CKPT_PATH="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/model.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ ERROR: Checkpoint not found at ${CKPT_PATH}"
    exit 1
fi

# Auto-detect training dataset from training_config.yaml (written by train.sh).
# This overrides the hardcoded dataset from mctd_dim_menu so the correct
# observation normalization stats and episode_len are used.
_TRAINING_CONFIG="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/training_config.yaml"
if [ -f "$_TRAINING_CONFIG" ]; then
    _DETECTED_DATASET=$(python3 -c "
import yaml, sys
with open('$_TRAINING_CONFIG') as f:
    d = yaml.safe_load(f)
print((d.get('dataset') or {}).get('config', ''))
" 2>/dev/null)
    if [ -n "$_DETECTED_DATASET" ] && [ "$_DETECTED_DATASET" != "None" ]; then
        if [ "$_DETECTED_DATASET" != "$SELECTED_DATASET" ]; then
            echo "  [config] Overriding dataset: $SELECTED_DATASET → $_DETECTED_DATASET (from training_config.yaml)"
        fi
        SELECTED_DATASET="$_DETECTED_DATASET"
    else
        echo "  [config] WARNING: training_config.yaml found but no dataset.config field. Using dim-menu default: $SELECTED_DATASET"
    fi
else
    echo "  [config] WARNING: No training_config.yaml for $SELECTED_MODEL_ID. Using dim-menu default: $SELECTED_DATASET"
    echo "           (Train with current train.sh to save this automatically)"
fi
echo ""

# ─────────────────────────────────────────────────────
# 3. Fixed parameters
# ─────────────────────────────────────────────────────
NUM_TASKS=3
NUM_SEEDS=1
START_TASK_IDX=3

echo "Configuration summary:"
echo "  Dataset    : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model      : $SELECTED_MODEL_ID"
echo "  Tasks      : $NUM_TASKS (start=$START_TASK_IDX)  Seeds: $NUM_SEEDS"
echo ""

# ─────────────────────────────────────────────────────
# 4. Execute Job Generator
# ─────────────────────────────────────────────────────
echo "Running: python3 scripts/generate_jobs_generalized.py --dataset $SELECTED_DATASET --model_id $SELECTED_MODEL_ID ..."
python3 scripts/generate_jobs_generalized.py \
    --dataset "$SELECTED_DATASET" \
    --model_id "$SELECTED_MODEL_ID" \
    --num_tasks "$NUM_TASKS" \
    --num_seeds "$NUM_SEEDS" \
    --start_task_id "$START_TASK_IDX"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Job generation failed!"
    exit 1
fi

echo ""
echo "Generation Complete."
echo ""
echo "====================================================="
echo "  Starting Job Execution via scripts/run_jobs.py"
echo "====================================================="
echo ""

python3 scripts/run_jobs.py 2>&1 | tee /tmp/mctd_run_jobs.log

JOB_EXIT_CODE=$?

echo ""
if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "====================================================="
    echo "  ✅ All Jobs Complete Successfully!"
    echo "====================================================="
else
    echo "====================================================="
    echo "  ❌ Jobs Failed with exit code: $JOB_EXIT_CODE"
    echo "====================================================="
    exit $JOB_EXIT_CODE
fi
