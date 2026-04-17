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

echo "Cleaning up existing benchmark job files..."
rm -f "$PROJECT_DIR"/jobs/*.json
echo "Job file cleanup complete"

echo "Resetting JAX cache..."
rm -rf ~/.jax_cache
mkdir -p ~/.jax_cache/xla_gpu_per_fusion_autotune_cache_dir
chmod -R 777 ~/.jax_cache
echo "JAX cache reset complete"
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
echo "Select rollouts per task:"
echo "  1) 50"
echo "  2) 20"
while true; do
    read -r -p "Choose [1-2] (default: 1): " _rollouts_input
    _rollouts_input="${_rollouts_input:-1}"
    case "$_rollouts_input" in
        1)
            ROLLOUTS_PER_TASK="50"
            break
            ;;
        2)
            ROLLOUTS_PER_TASK="20"
            break
            ;;
        *)
            echo "Please enter 1 or 2."
            ;;
    esac
done

RESULTS_RUN_DIR="benchmark_results/${SELECTED_MODEL_ID}/run_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$PROJECT_DIR/$RESULTS_RUN_DIR"

echo ""
echo "Configuration summary:"
echo "  Dataset           : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model             : $SELECTED_MODEL_ID"
echo "  Repeats           : $NUM_REPEATS"
echo "  Tasks             : $NUM_TASKS"
echo "  Rollouts per task : $ROLLOUTS_PER_TASK"
echo "  Results dir       : $RESULTS_RUN_DIR"
echo ""

echo "Generating benchmark jobs..."
python3 "$PROJECT_DIR/scripts/generate_benchmark_jobs.py" \
    --dataset "$SELECTED_DATASET" \
    --model_id "$SELECTED_MODEL_ID" \
    --num_tasks "$NUM_TASKS" \
    --num_repeats "$NUM_REPEATS" \
    --rollouts_per_task "$ROLLOUTS_PER_TASK" \
    --results_dir "$RESULTS_RUN_DIR"

echo ""
echo "====================================================="
echo "  Running benchmark jobs"
echo "====================================================="
echo ""

export AVAILABLE_GPUS
PIPELINE_LOG="/tmp/mctd_eval_all_${SELECTED_MODEL_ID}.log"
PIPELINE_PID_FILE="/tmp/mctd_eval_all_${SELECTED_MODEL_ID}.pid"
nohup setsid bash "$PROJECT_DIR/scripts/run_benchmark_pipeline.sh" \
    "$RESULTS_RUN_DIR" \
    "$NUM_REPEATS" \
    "$NUM_TASKS" \
    "$ROLLOUTS_PER_TASK" > "$PIPELINE_LOG" 2>&1 &
PIPELINE_PID=$!
echo "$PIPELINE_PID" > "$PIPELINE_PID_FILE"

echo "Jobs launched in background (PID: $PIPELINE_PID)"
echo "  Log: $PIPELINE_LOG"
echo "  Monitor: tail -f $PIPELINE_LOG"
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
echo ""
echo "====================================================="
echo "  Benchmark Pipeline Running in Background"
echo "====================================================="
