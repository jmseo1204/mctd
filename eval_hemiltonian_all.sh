#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DATASET_DIR="$PROJECT_DIR/configurations/dataset"
LOCAL_OGBENCH_DIR="$(cd "$PROJECT_DIR/../ogbench" 2>/dev/null && pwd || true)"

MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

OUTPUT_DOWNLOADED_DIR="$MCTD_DOWNLOADED_DIR"
NUM_TASKS=5
NUM_REPEATS=3
WAYPOINT_TOP_N=10
DEFAULT_OVERRIDE_PATH="configurations/task_overrides/antmaze_giant_waypoints_example.yaml"

echo "===================================================="
echo "  MCTD Hamiltonian Benchmark Launcher"
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
    echo "ERROR: Hamiltonian benchmark requires local ogbench checkout at $PROJECT_DIR/../ogbench"
    exit 1
fi
echo "Using local ogbench source for benchmark jobs: $LOCAL_OGBENCH_DIR"
echo ""

echo "Cleaning up existing benchmark job files..."
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
echo "Enter number of eval repeats (positive integer, default: ${NUM_REPEATS}):"
while true; do
    read -r -p "Eval repeats: " _repeat_input
    _repeat_input="${_repeat_input:-$NUM_REPEATS}"
    if [[ "$_repeat_input" =~ ^[1-9][0-9]*$ ]]; then
        NUM_REPEATS="$_repeat_input"
        break
    fi
    echo "Please enter a positive integer."
done

echo ""
read -r -p "Task override path [Enter=${DEFAULT_OVERRIDE_PATH}]: " TASK_OVERRIDE_PATH
TASK_OVERRIDE_PATH="${TASK_OVERRIDE_PATH:-${DEFAULT_OVERRIDE_PATH}}"
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

echo ""
echo "Enter top N ranked waypoint groups per task (positive integer, default: ${WAYPOINT_TOP_N}):"
while true; do
    read -r -p "Top N waypoint groups: " _topn_input
    _topn_input="${_topn_input:-$WAYPOINT_TOP_N}"
    if [[ "$_topn_input" =~ ^[1-9][0-9]*$ ]]; then
        WAYPOINT_TOP_N="$_topn_input"
        break
    fi
    echo "Please enter a positive integer."
done

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_FILE_PREFIX="${SELECTED_DATASET}_hamiltonian_${RUN_TIMESTAMP}"
RESULTS_RUN_DIR="benchmark_results/${SELECTED_MODEL_ID}/${RESULTS_FILE_PREFIX}"
SUMMARY_JSON_REL="${RESULTS_RUN_DIR}/${RESULTS_FILE_PREFIX}_summary.json"
PLANNING_CONFIG_SRC="$PROJECT_DIR/configurations/algorithm/df_planning.yaml"
PLANNING_CONFIG_REL="${RESULTS_RUN_DIR}/df_planning.yaml"
TASK_OVERRIDE_SNAPSHOT_REL="${RESULTS_RUN_DIR}/$(basename "$TASK_OVERRIDE_PATH")"
mkdir -p "$PROJECT_DIR/$RESULTS_RUN_DIR"
cp -f "$PLANNING_CONFIG_SRC" "$PROJECT_DIR/$PLANNING_CONFIG_REL"
cp -f "$PROJECT_DIR/$TASK_OVERRIDE_PATH" "$PROJECT_DIR/$TASK_OVERRIDE_SNAPSHOT_REL"

echo ""
echo "Configuration summary:"
echo "  Dataset             : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model               : $SELECTED_MODEL_ID"
echo "  Repeats             : $NUM_REPEATS"
echo "  Tasks               : $NUM_TASKS"
echo "  Top N groups / task : $WAYPOINT_TOP_N"
echo "  Results dir         : $RESULTS_RUN_DIR"
echo "  Summary JSON        : $SUMMARY_JSON_REL"
echo "  Planning config     : $PLANNING_CONFIG_REL"
echo "  Task override       : $TASK_OVERRIDE_SNAPSHOT_REL"
echo ""

echo "Generating Hamiltonian benchmark jobs..."
python3 "$PROJECT_DIR/scripts/generate_hamiltonian_benchmark_jobs.py" \
    --dataset "$SELECTED_DATASET" \
    --model_id "$SELECTED_MODEL_ID" \
    --num_tasks "$NUM_TASKS" \
    --num_repeats "$NUM_REPEATS" \
    --waypoint_top_n "$WAYPOINT_TOP_N" \
    --task_override_path "$TASK_OVERRIDE_SNAPSHOT_REL" \
    --planning_config_snapshot "$PLANNING_CONFIG_REL" \
    --results_dir "$RESULTS_RUN_DIR" \
    --results_file_prefix "$RESULTS_FILE_PREFIX"

echo ""
echo "====================================================="
echo "  Running Hamiltonian Benchmark Jobs"
echo "====================================================="
echo ""

export AVAILABLE_GPUS
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/eval_hemiltonian_all_${SELECTED_DATASET}_${RUN_TIMESTAMP}.log"
SCHEDULER_LOG="$LOG_DIR/run_${RUN_TIMESTAMP}.log"
PIPELINE_PID_FILE="$LOG_DIR/eval_hemiltonian_all_${SELECTED_MODEL_ID}_${RUN_TIMESTAMP}.pid"
nohup setsid bash "$PROJECT_DIR/scripts/run_hamiltonian_benchmark_pipeline.sh" \
    "$RESULTS_RUN_DIR" \
    "$NUM_REPEATS" \
    "$NUM_TASKS" \
    "$WAYPOINT_TOP_N" \
    "$SUMMARY_JSON_REL" \
    "$SELECTED_DATASET" \
    "$RUN_TIMESTAMP" \
    "$TASK_OVERRIDE_SNAPSHOT_REL" > "$PIPELINE_LOG" 2>&1 &
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
echo "WandB group: HBENCH-$SELECTED_MODEL_ID"
echo "Results directory: $PROJECT_DIR/$RESULTS_RUN_DIR"
echo "Summary JSON: $PROJECT_DIR/$SUMMARY_JSON_REL"
echo "Planning config snapshot: $PROJECT_DIR/$PLANNING_CONFIG_REL"
echo "Task override snapshot: $PROJECT_DIR/$TASK_OVERRIDE_SNAPSHOT_REL"
echo ""
echo "====================================================="
echo "  Hamiltonian Benchmark Pipeline Running in Background"
echo "====================================================="
