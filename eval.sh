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

OUTPUT_DOWNLOADED_DIR="$MCTD_DOWNLOADED_DIR"

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

echo "Job queue cleanup will run after dataset detection."

echo "Resetting JAX cache..."
rm -rf ~/.jax_cache
mkdir -p ~/.jax_cache/xla_gpu_per_fusion_autotune_cache_dir
chmod -R 777 ~/.jax_cache
echo "✓ JAX cache reset complete"
echo ""

echo "Preparing outputs directory permissions..."
if ! mctd_prepare_output_permissions; then
    exit 1
fi
echo "✓ Outputs directory ready: $MCTD_OUTPUT_MOUNT_DIR"
echo ""

# GPU selection + availability check (after container cleanup to avoid false positives)
mctd_select_gpus
mctd_check_gpu_availability

# ─────────────────────────────────────────────────────
# 1. Scan ALL checkpoints and let user select by index
# ─────────────────────────────────────────────────────
echo "Scanning all checkpoints..."
mctd_scan_ckpts 0   # 0 = no obs_dim filter; show all

if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
    echo "❌ ERROR: No checkpoints found in $MCTD_OUTPUT_MOUNT_DIR"
    exit 1
fi

mctd_ckpt_menu "--no-fresh" || exit 1

# model_id is populated directly by scanner — no fallback needed
SELECTED_MODEL_ID="${MCTD_SELECTED_MODEL_ID:-}"
if [ -z "$SELECTED_MODEL_ID" ]; then
    echo "❌ ERROR: Could not determine model_id for selected checkpoint." >&2
    exit 1
fi
mctd_ensure_eval_symlink "$MCTD_SELECTED_CKPT" "$SELECTED_MODEL_ID"
echo "✓ Selected model: $SELECTED_MODEL_ID"

CKPT_PATH="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/model.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ ERROR: Checkpoint not found at ${CKPT_PATH}"
    exit 1
fi

# ─────────────────────────────────────────────────────
# 2. Auto-detect dataset and obs_dim_indices from checkpoint metadata
# Priority: training_config.yaml > scanner metadata > error
# ─────────────────────────────────────────────────────
SELECTED_DATASET="${MCTD_SELECTED_DATASET:-unknown}"
OBS_DIM_INDICES="${MCTD_SELECTED_OBS_DIM_INDICES:-unknown}"

_TRAINING_CONFIG="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/training_config.yaml"
if [ -f "$_TRAINING_CONFIG" ]; then
    _DETECTED=$(python3 -c "
import yaml, json, sys
with open('$_TRAINING_CONFIG') as f:
    d = yaml.safe_load(f)
ds = (d.get('dataset') or {}).get('config', '')
obs_idx = (d.get('algorithm') or {}).get('obs_dim_indices')
print(ds)
print(json.dumps(obs_idx) if obs_idx else '')
" 2>/dev/null)
    _DETECTED_DATASET=$(echo "$_DETECTED" | sed -n '1p')
    _DETECTED_INDICES=$(echo "$_DETECTED" | sed -n '2p')
    [ -n "$_DETECTED_DATASET" ] && [ "$_DETECTED_DATASET" != "None" ] && SELECTED_DATASET="$_DETECTED_DATASET"
    [ -n "$_DETECTED_INDICES" ] && [ "$_DETECTED_INDICES" != "None" ] && OBS_DIM_INDICES="$_DETECTED_INDICES"
    echo "  [config] Loaded metadata from training_config.yaml: dataset=$SELECTED_DATASET obs_dim_indices=$OBS_DIM_INDICES"
else
    echo "  [config] No training_config.yaml for $SELECTED_MODEL_ID — using scanner metadata."
    echo "           dataset=$SELECTED_DATASET  obs_dim_indices=$OBS_DIM_INDICES"
fi

# Derive count for display only
STATE_DIM=$(python3 -c "import json; v='$OBS_DIM_INDICES'; print(len(json.loads(v)) if v not in ('unknown','') else '?')" 2>/dev/null || echo "?")

# If dataset still unknown, fail with guidance
if [ "$SELECTED_DATASET" = "unknown" ] || [ -z "$SELECTED_DATASET" ]; then
    echo "❌ ERROR: Could not determine training dataset for $SELECTED_MODEL_ID." >&2
    echo "   This is a legacy checkpoint without training_config.yaml." >&2
    echo "   Create $MCTD_EVAL_BASE/$SELECTED_MODEL_ID/training_config.yaml with dataset.config field." >&2
    exit 1
fi

SELECTED_DATASET="$(mctd_normalize_dataset_name "$SELECTED_DATASET" "$CONFIG_DATASET_DIR")"
DATASET_YAML="${CONFIG_DATASET_DIR}/${SELECTED_DATASET}.yaml"
if [ ! -f "$DATASET_YAML" ]; then
    echo "❌ ERROR: Dataset config not found: ${DATASET_YAML}"
    exit 1
fi
JOBS_DIR_REL="jobs/${SELECTED_DATASET}"
JOBS_DIR_ABS="$PROJECT_DIR/$JOBS_DIR_REL"

echo "Cleaning up existing evaluation job files for dataset queue: $JOBS_DIR_REL"
mkdir -p "$JOBS_DIR_ABS"
rm -f "$JOBS_DIR_ABS"/*.json
echo "✓ Job files cleanup complete"
echo ""

# ─────────────────────────────────────────────────────
# 3. Fixed parameters
# ─────────────────────────────────────────────────────
NUM_TASKS=5
NUM_SEEDS=1
START_TASK_IDX=2

echo "Configuration summary:"
echo "  Dataset    : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model      : $SELECTED_MODEL_ID"
echo "  Tasks      : $NUM_TASKS (start=$START_TASK_IDX)  Seeds: $NUM_SEEDS"
echo "  Jobs queue : $JOBS_DIR_REL"
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
    --start_task_id "$START_TASK_IDX" \
    --jobs_dir "$JOBS_DIR_REL"

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

export AVAILABLE_GPUS
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SCHEDULER_LOG="$PROJECT_DIR/logs/run_${SELECTED_DATASET}_${RUN_TIMESTAMP}.log"
nohup env MCTD_RUN_TIMESTAMP="${SELECTED_DATASET}_${RUN_TIMESTAMP}" MCTD_JOBS_DIR="$JOBS_DIR_REL" PYTHONUNBUFFERED=1 python3 scripts/run_jobs.py > /tmp/mctd_run_jobs.log 2>&1 &
RUN_JOBS_PID=$!

echo "✓ Jobs launched in background (PID: $RUN_JOBS_PID)"
echo "  Log: /tmp/mctd_run_jobs.log"
echo "  Monitor: tail -f /tmp/mctd_run_jobs.log"
echo "  Scheduler log: $SCHEDULER_LOG"
echo "  Monitor scheduler: tail -f $SCHEDULER_LOG"
echo "  Containers: docker ps --filter 'name=exp_gpu'"
echo ""
echo "====================================================="
echo "  ✅ Jobs Running in Background"
echo "====================================================="
