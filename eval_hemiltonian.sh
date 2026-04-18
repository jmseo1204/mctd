#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DATASET_DIR="$PROJECT_DIR/configurations/dataset"

MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

OUTPUT_DOWNLOADED_DIR="$MCTD_DOWNLOADED_DIR"
NUM_TASKS=5
NUM_SEEDS=1
START_TASK_IDX=1
DEFAULT_OVERRIDE_PATH="configurations/task_overrides/antmaze_giant_waypoints_example.yaml"

echo "===================================================="
echo "  MCTD Hamiltonian Evaluation Launcher"
echo "===================================================="

echo "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running"
    exit 1
fi
echo "Docker is available and running"
echo ""

echo "Cleaning up existing job files..."
rm -f "$PROJECT_DIR"/jobs/*.json
echo "Job file cleanup complete"

echo "Resetting JAX cache..."
rm -rf ~/.jax_cache
mkdir -p ~/.jax_cache/xla_gpu_per_fusion_autotune_cache_dir
chmod -R 777 ~/.jax_cache
echo "JAX cache reset complete"
echo ""

echo "Preparing outputs directory permissions..."
mctd_prepare_output_permissions
echo "Outputs directory ready: $MCTD_OUTPUT_MOUNT_DIR"
echo ""

mctd_select_gpus
mctd_check_gpu_availability

echo "Scanning all checkpoints..."
mctd_scan_ckpts 0
if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in $MCTD_OUTPUT_MOUNT_DIR"
    exit 1
fi

mctd_ckpt_menu "--no-fresh" || exit 1

SELECTED_MODEL_ID="${MCTD_SELECTED_MODEL_ID:-}"
if [ -z "$SELECTED_MODEL_ID" ]; then
    echo "ERROR: Could not determine model_id for selected checkpoint." >&2
    exit 1
fi
mctd_ensure_eval_symlink "$MCTD_SELECTED_CKPT" "$SELECTED_MODEL_ID"
echo "Selected model: $SELECTED_MODEL_ID"

CKPT_PATH="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/model.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at ${CKPT_PATH}"
    exit 1
fi

SELECTED_DATASET="${MCTD_SELECTED_DATASET:-unknown}"
OBS_DIM_INDICES="${MCTD_SELECTED_OBS_DIM_INDICES:-unknown}"
_TRAINING_CONFIG="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/training_config.yaml"
if [ -f "$_TRAINING_CONFIG" ]; then
    _DETECTED=$(python3 -c "
import yaml, json
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

STATE_DIM=$(python3 -c "import json; v='$OBS_DIM_INDICES'; print(len(json.loads(v)) if v not in ('unknown','') else '?')" 2>/dev/null || echo "?")

if [ "$SELECTED_DATASET" = "unknown" ] || [ -z "$SELECTED_DATASET" ]; then
    echo "ERROR: Could not determine training dataset for $SELECTED_MODEL_ID." >&2
    exit 1
fi

SELECTED_DATASET="$(mctd_normalize_dataset_name "$SELECTED_DATASET" "$CONFIG_DATASET_DIR")"
DATASET_YAML="${CONFIG_DATASET_DIR}/${SELECTED_DATASET}.yaml"
if [ ! -f "$DATASET_YAML" ]; then
    echo "ERROR: Dataset config not found: ${DATASET_YAML}"
    exit 1
fi

echo ""
read -r -p "Task override path [Enter=${DEFAULT_OVERRIDE_PATH}, none=disable]: " TASK_OVERRIDE_PATH
TASK_OVERRIDE_PATH="${TASK_OVERRIDE_PATH:-${DEFAULT_OVERRIDE_PATH}}"

WAYPOINT_GROUP_IDX=""
if [ "$TASK_OVERRIDE_PATH" != "none" ]; then
    if [[ "$TASK_OVERRIDE_PATH" = /* ]]; then
        if [[ "$TASK_OVERRIDE_PATH" == "${PROJECT_DIR}/"* ]]; then
            TASK_OVERRIDE_PATH="${TASK_OVERRIDE_PATH#${PROJECT_DIR}/}"
        else
            echo "ERROR: Task override path must be inside ${PROJECT_DIR}"
            exit 1
        fi
    fi
    if [ ! -f "${PROJECT_DIR}/${TASK_OVERRIDE_PATH}" ]; then
        echo "ERROR: Task override file not found: ${PROJECT_DIR}/${TASK_OVERRIDE_PATH}"
        exit 1
    fi
    read -r -p "Waypoint group index [Enter=active/default]: " WAYPOINT_GROUP_IDX
    if [ -n "$WAYPOINT_GROUP_IDX" ] && ! [[ "$WAYPOINT_GROUP_IDX" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Waypoint group index must be a non-negative integer."
        exit 1
    fi
fi

echo ""
echo "Configuration summary:"
echo "  Dataset    : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model      : $SELECTED_MODEL_ID"
echo "  Tasks      : $NUM_TASKS (start=$START_TASK_IDX)  Seeds: $NUM_SEEDS"
echo "  Planner    : multi_tree_hemiltonian=true"
if [ "$TASK_OVERRIDE_PATH" = "none" ]; then
    echo "  Override   : none"
else
    echo "  Override   : $TASK_OVERRIDE_PATH"
    if [ -n "$WAYPOINT_GROUP_IDX" ]; then
        echo "  Group idx  : $WAYPOINT_GROUP_IDX"
    fi
fi
echo ""

GEN_ARGS=(
    --dataset "$SELECTED_DATASET"
    --model_id "$SELECTED_MODEL_ID"
    --num_tasks "$NUM_TASKS"
    --num_seeds "$NUM_SEEDS"
    --start_task_id "$START_TASK_IDX"
    --multi_tree_hemiltonian
)
if [ "$TASK_OVERRIDE_PATH" != "none" ]; then
    GEN_ARGS+=(--task_override_path "$TASK_OVERRIDE_PATH")
    if [ -n "$WAYPOINT_GROUP_IDX" ]; then
        GEN_ARGS+=(--task_override_waypoint_group_idx "$WAYPOINT_GROUP_IDX")
    fi
fi

echo "Generating evaluation jobs..."
python3 "$PROJECT_DIR/scripts/generate_jobs_generalized.py" "${GEN_ARGS[@]}"

echo ""
echo "====================================================="
echo "  Starting Job Execution via scripts/run_jobs.py"
echo "====================================================="
echo ""

export AVAILABLE_GPUS
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SCHEDULER_LOG="$PROJECT_DIR/logs/run_${RUN_TIMESTAMP}.log"
nohup env MCTD_RUN_TIMESTAMP="$RUN_TIMESTAMP" PYTHONUNBUFFERED=1 python3 "$PROJECT_DIR/scripts/run_jobs.py" > /tmp/mctd_run_jobs.log 2>&1 &
RUN_JOBS_PID=$!

echo "Jobs launched in background (PID: $RUN_JOBS_PID)"
echo "  Log: /tmp/mctd_run_jobs.log"
echo "  Monitor: tail -f /tmp/mctd_run_jobs.log"
echo "  Scheduler log: $SCHEDULER_LOG"
echo "  Monitor scheduler: tail -f $SCHEDULER_LOG"
echo "  Containers: docker ps --filter 'name=exp_gpu'"
