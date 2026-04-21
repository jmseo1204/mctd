#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DATASET_DIR="$PROJECT_DIR/configurations/dataset"
LOCAL_OGBENCH_DIR="$(cd "$PROJECT_DIR/../ogbench" 2>/dev/null && pwd || true)"

MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

OUTPUT_DOWNLOADED_DIR="$MCTD_DOWNLOADED_DIR"
NUM_REPEATS=3
NUM_TASKS=5
ROLLOUTS_PER_TASK=50

echo "===================================================="
echo "  MCTD Single-Checkpoint Benchmark Launcher"
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

if [ -z "$LOCAL_OGBENCH_DIR" ] || [ ! -d "$LOCAL_OGBENCH_DIR" ]; then
    echo "ERROR: Benchmark pipeline requires local ogbench checkout at $PROJECT_DIR/../ogbench"
    exit 1
fi
echo "Using local ogbench source for benchmark jobs: $LOCAL_OGBENCH_DIR"
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

SELECTED_CKPT_HOST="$MCTD_SELECTED_CKPT"
SELECTED_CKPT_REAL="$(realpath "$MCTD_SELECTED_CKPT" 2>/dev/null || true)"
if [ ! -f "$SELECTED_CKPT_HOST" ] && { [ -z "$SELECTED_CKPT_REAL" ] || [ ! -f "$SELECTED_CKPT_REAL" ]; }; then
    echo "ERROR: Selected checkpoint is not a readable file: ${MCTD_SELECTED_CKPT}" >&2
    exit 1
fi

CONTAINER_OUTPUT_ROOT="/home/${MCTD_DOCKER_USER}/mctd_outputs"
CKPT_HOST_FOR_LOAD=""
for _candidate in "$SELECTED_CKPT_HOST" "$SELECTED_CKPT_REAL"; do
    if [ -n "$_candidate" ] && [[ "$_candidate" == "$MCTD_OUTPUT_MOUNT_DIR/"* ]]; then
        CKPT_HOST_FOR_LOAD="$_candidate"
        break
    fi
done
if [ -z "$CKPT_HOST_FOR_LOAD" ]; then
    echo "ERROR: Selected checkpoint is outside the Docker outputs mount: ${MCTD_SELECTED_CKPT}" >&2
    exit 1
fi
CKPT_LOAD_PATH="${CKPT_HOST_FOR_LOAD/#$MCTD_OUTPUT_MOUNT_DIR/$CONTAINER_OUTPUT_ROOT}"

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

JOBS_DIR_REL="jobs/${SELECTED_DATASET}"
JOBS_DIR_ABS="$PROJECT_DIR/$JOBS_DIR_REL"

echo "Cleaning up existing benchmark job files for dataset queue: $JOBS_DIR_REL"
mkdir -p "$JOBS_DIR_ABS"
rm -f "$JOBS_DIR_ABS"/*.json
echo "Dataset queue cleanup complete"

echo ""
echo "Enter number of evaluation repeats (positive integer, default: ${NUM_REPEATS}):"
while true; do
    read -r -p "Evaluation repeats: " _repeats_input
    _repeats_input="${_repeats_input:-$NUM_REPEATS}"
    if [[ "$_repeats_input" =~ ^[1-9][0-9]*$ ]]; then
        NUM_REPEATS="$_repeats_input"
        break
    fi
    echo "Please enter a positive integer."
done

echo ""
echo "Enter rollouts per task (positive integer, default: ${ROLLOUTS_PER_TASK}):"
while true; do
    read -r -p "Rollouts per task: " _rollouts_input
    _rollouts_input="${_rollouts_input:-$ROLLOUTS_PER_TASK}"
    if [[ "$_rollouts_input" =~ ^[1-9][0-9]*$ ]]; then
        ROLLOUTS_PER_TASK="$_rollouts_input"
        break
    fi
    echo "Please enter a positive integer."
done

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_FILE_PREFIX="${SELECTED_DATASET}_${RUN_TIMESTAMP}"
RESULTS_RUN_DIR="benchmark_results/${SELECTED_MODEL_ID}/${RESULTS_FILE_PREFIX}"
SUMMARY_JSON_REL="${RESULTS_RUN_DIR}/${RESULTS_FILE_PREFIX}_summary.json"
PLANNING_CONFIG_SRC="$PROJECT_DIR/configurations/algorithm/df_planning.yaml"
PLANNING_CONFIG_REL="${RESULTS_RUN_DIR}/df_planning.yaml"
mkdir -p "$PROJECT_DIR/$RESULTS_RUN_DIR"
cp -f "$PLANNING_CONFIG_SRC" "$PROJECT_DIR/$PLANNING_CONFIG_REL"

echo ""
echo "Configuration summary:"
echo "  Dataset           : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model             : $SELECTED_MODEL_ID"
echo "  Checkpoint path   : $CKPT_LOAD_PATH"
echo "  Repeats           : $NUM_REPEATS"
echo "  Tasks             : $NUM_TASKS"
echo "  Rollouts per task : $ROLLOUTS_PER_TASK"
echo "  Jobs queue        : $JOBS_DIR_REL"
echo "  Results dir       : $RESULTS_RUN_DIR"
echo "  Task result files : $RESULTS_RUN_DIR/${RESULTS_FILE_PREFIX}_repeat_<repeat>_task_<task>.json"
echo "  Summary JSON      : $SUMMARY_JSON_REL"
echo "  Planning config   : $PLANNING_CONFIG_REL"
echo ""

echo "Generating benchmark jobs..."
python3 "$PROJECT_DIR/scripts/generate_benchmark_jobs.py" \
    --dataset "$SELECTED_DATASET" \
    --model_id "$SELECTED_MODEL_ID" \
    --load_path "$CKPT_LOAD_PATH" \
    --num_tasks "$NUM_TASKS" \
    --num_repeats "$NUM_REPEATS" \
    --rollouts_per_task "$ROLLOUTS_PER_TASK" \
    --jobs_dir "$JOBS_DIR_REL" \
    --planning_config_snapshot "$PLANNING_CONFIG_REL" \
    --results_dir "$RESULTS_RUN_DIR" \
    --results_file_prefix "$RESULTS_FILE_PREFIX"

echo ""
echo "====================================================="
echo "  Running benchmark jobs"
echo "====================================================="
echo ""

export AVAILABLE_GPUS
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/eval_all_${SELECTED_DATASET}_${RUN_TIMESTAMP}.log"
SCHEDULER_LOG="$LOG_DIR/run_${SELECTED_DATASET}_${RUN_TIMESTAMP}.log"
PIPELINE_PID_FILE="$LOG_DIR/eval_all_${SELECTED_MODEL_ID}_${RUN_TIMESTAMP}.pid"
nohup setsid bash "$PROJECT_DIR/scripts/run_benchmark_pipeline.sh" \
    "$RESULTS_RUN_DIR" \
    "$NUM_REPEATS" \
    "$NUM_TASKS" \
    "$ROLLOUTS_PER_TASK" \
    "$SUMMARY_JSON_REL" \
    "$SELECTED_DATASET" \
    "$RUN_TIMESTAMP" \
    "$JOBS_DIR_REL" > "$PIPELINE_LOG" 2>&1 &
PIPELINE_PID=$!
echo "$PIPELINE_PID" > "$PIPELINE_PID_FILE"

echo "Jobs launched in background (PID: $PIPELINE_PID)"
echo "  Primary log: $SCHEDULER_LOG"
echo "  Monitor: tail -f $SCHEDULER_LOG"
echo "  Pipeline log: $PIPELINE_LOG"
echo "  Monitor pipeline: tail -f $PIPELINE_LOG"
echo "  Containers: docker ps --filter 'name=exp_gpu'"
echo "  PID file: $PIPELINE_PID_FILE"
echo ""
echo "Stop guide:"
echo "  Stop scheduler group : kill -- -\$(cat \"$PIPELINE_PID_FILE\")"
echo "  Check scheduler pid  : ps -fp \$(cat \"$PIPELINE_PID_FILE\")"
echo "  Stop running containers too:"
echo "    docker ps --filter 'name=exp_gpu' -q | xargs -r docker rm -f"
echo ""
echo "WandB group: BENCH-$SELECTED_MODEL_ID"
echo "Results directory: $PROJECT_DIR/$RESULTS_RUN_DIR"
echo "Task result files: $PROJECT_DIR/$RESULTS_RUN_DIR/${RESULTS_FILE_PREFIX}_repeat_<repeat>_task_<task>.json"
echo "Summary JSON: $PROJECT_DIR/$SUMMARY_JSON_REL"
echo "Planning config snapshot: $PROJECT_DIR/$PLANNING_CONFIG_REL"
echo ""
echo "====================================================="
echo "  Benchmark Pipeline Running in Background"
echo "====================================================="
