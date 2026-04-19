#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DATASET_DIR="$PROJECT_DIR/configurations/dataset"
LOCAL_OGBENCH_DIR="$(cd "$PROJECT_DIR/../ogbench" 2>/dev/null && pwd || true)"
PLANNING_CONFIG_SRC="$PROJECT_DIR/configurations/algorithm/df_planning.yaml"

MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"
DOCKER_PROJECT="/home/${DOCKER_USER}/mctd"

OUTPUT_DOWNLOADED_DIR="$MCTD_DOWNLOADED_DIR"
NUM_TASKS=5
NUM_REPEATS=3
WAYPOINT_TOP_N=10

read_yaml_value() {
    local yaml_path="$1"
    local dotted_key="$2"
    python3 - "$yaml_path" "$dotted_key" <<'PY'
import sys
import yaml

yaml_path, dotted_key = sys.argv[1], sys.argv[2]
with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
value = data
for token in dotted_key.split("."):
    if not isinstance(value, dict) or token not in value:
        value = None
        break
    value = value[token]
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

normalize_env_key() {
    local env_id="$1"
    python3 - "$env_id" <<'PY'
import re
import sys

env_id = sys.argv[1].strip()
env_key = re.sub(r"[^0-9A-Za-z]+", "_", env_id).strip("_")
env_key = re.sub(r"_v[0-9]+$", "", env_key)
print(env_key)
PY
}

resolve_repo_path() {
    local rel_or_abs="$1"
    python3 - "$PROJECT_DIR" "$rel_or_abs" <<'PY'
import os
import sys

repo_root, path_raw = sys.argv[1], sys.argv[2]
expanded = os.path.expanduser(path_raw)
if not os.path.isabs(expanded):
    expanded = os.path.join(repo_root, expanded)
print(os.path.realpath(expanded))
PY
}

to_repo_relative_path() {
    local abs_path="$1"
    python3 - "$PROJECT_DIR" "$abs_path" <<'PY'
import os
import sys

repo_root, abs_path = sys.argv[1], sys.argv[2]
print(os.path.relpath(os.path.realpath(abs_path), repo_root))
PY
}

resolve_graph_cache_host_dir() {
    local cache_dir_raw="$1"
    python3 - "$PROJECT_DIR" "$DOCKER_USER" "$cache_dir_raw" <<'PY'
import os
import sys

repo_root, docker_user, cache_dir_raw = sys.argv[1], sys.argv[2], sys.argv[3]
if cache_dir_raw in ("", "None", "null"):
    cache_dir_raw = "~/.ogbench/data"
expanded = os.path.expanduser(cache_dir_raw)
docker_default = f"/home/{docker_user}/.ogbench/data"
host_default = os.path.realpath(os.path.join(os.path.dirname(repo_root), "ogbench_data"))
if expanded == docker_default or expanded.startswith(docker_default + os.sep):
    suffix = expanded[len(docker_default):].lstrip(os.sep)
    resolved = os.path.join(host_default, suffix)
elif not os.path.isabs(expanded):
    resolved = os.path.join(repo_root, expanded)
else:
    resolved = expanded
print(os.path.realpath(resolved))
PY
}

resolve_graph_cache_host_path() {
    local cache_path_raw="$1"
    python3 - "$PROJECT_DIR" "$DOCKER_USER" "$cache_path_raw" <<'PY'
import os
import sys

repo_root, docker_user, cache_path_raw = sys.argv[1], sys.argv[2], sys.argv[3]
raw = str(cache_path_raw).strip()
if raw in ("", "None", "null"):
    print("")
    raise SystemExit(0)
docker_default = f"/home/{docker_user}/.ogbench/data"
host_default = os.path.realpath(os.path.join(os.path.dirname(repo_root), "ogbench_data"))
if raw == "~/.ogbench/data" or raw.startswith("~/.ogbench/data" + os.sep):
    suffix = raw[len("~/.ogbench/data"):].lstrip(os.sep)
    resolved = os.path.join(host_default, suffix)
else:
    expanded = os.path.expanduser(raw)
    if expanded == docker_default or expanded.startswith(docker_default + os.sep):
        suffix = expanded[len(docker_default):].lstrip(os.sep)
        resolved = os.path.join(host_default, suffix)
    elif not os.path.isabs(expanded):
        resolved = os.path.join(repo_root, expanded)
    else:
        resolved = expanded
print(os.path.realpath(resolved))
PY
}

to_graph_cache_container_path() {
    local cache_host_path="$1"
    python3 - "$PROJECT_DIR" "$cache_host_path" <<'PY'
import os
import sys

repo_root, cache_host_path = sys.argv[1], sys.argv[2]
host_default = os.path.realpath(os.path.join(os.path.dirname(repo_root), "ogbench_data"))
cache_host_path = os.path.realpath(cache_host_path)
if cache_host_path == host_default or cache_host_path.startswith(host_default + os.sep):
    suffix = os.path.relpath(cache_host_path, host_default)
    print(os.path.join("~/.ogbench/data", suffix))
else:
    print(cache_host_path)
PY
}

find_latest_matching_graph_cache() {
    local cache_dir="$1"
    local dataset="$2"
    local sample_ratio="$3"
    local edge_radius="$4"
    local seed="$5"
    python3 - "$cache_dir" "$dataset" "$sample_ratio" "$edge_radius" "$seed" <<'PY'
import glob
import os
import sys

cache_dir, dataset, sample_ratio, edge_radius, seed = sys.argv[1:]

def _float_token(value: str) -> str:
    return f"{float(value):g}".replace("-", "m")

pattern = os.path.join(
    cache_dir,
    f"{dataset}_sampled_graph_"
    f"r{_float_token(sample_ratio)}_"
    f"seed{int(seed)}_"
    f"rad{_float_token(edge_radius)}*.pkl",
)
matches = [os.path.realpath(path) for path in glob.glob(pattern)]
matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
print(matches[0] if matches else "")
PY
}

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

ENV_ID="$(read_yaml_value "$DATASET_YAML" "env_id")"
if [ -z "$ENV_ID" ]; then
    echo "ERROR: Could not determine env_id from ${DATASET_YAML}"
    exit 1
fi
ENV_KEY="$(normalize_env_key "$ENV_ID")"
DEFAULT_FEASIBLE_PATH="configurations/task_overrides/${ENV_KEY}_feasible_points.yaml"
DEFAULT_OVERRIDE_PATH="configurations/task_overrides/${ENV_KEY}_waypoints.yaml"

CONFIG_TASK_OVERRIDE_PATH="$(read_yaml_value "$PLANNING_CONFIG_SRC" "task_override_path")"
TASK_OVERRIDE_PATH="${CONFIG_TASK_OVERRIDE_PATH:-$DEFAULT_OVERRIDE_PATH}"
if [ "$TASK_OVERRIDE_PATH" = "null" ] || [ "$TASK_OVERRIDE_PATH" = "none" ] || [ "$TASK_OVERRIDE_PATH" = "None" ]; then
    TASK_OVERRIDE_PATH="$DEFAULT_OVERRIDE_PATH"
fi
if [[ "$TASK_OVERRIDE_PATH" = /* ]]; then
    if [[ "$TASK_OVERRIDE_PATH" == "${PROJECT_DIR}/"* ]]; then
        TASK_OVERRIDE_PATH="${TASK_OVERRIDE_PATH#${PROJECT_DIR}/}"
    else
        echo "ERROR: task_override_path in ${PLANNING_CONFIG_SRC} must be inside ${PROJECT_DIR}"
        exit 1
    fi
fi
FEASIBLE_PATH="$DEFAULT_FEASIBLE_PATH"
if [ ! -f "${PROJECT_DIR}/${FEASIBLE_PATH}" ]; then
    echo "ERROR: Feasible points file not found: ${PROJECT_DIR}/${FEASIBLE_PATH}"
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

DEFAULT_OGBENCH_RESET_PERTURB="$(read_yaml_value "$PLANNING_CONFIG_SRC" "ogbench_enable_reset_perturb")"
if [ -z "$DEFAULT_OGBENCH_RESET_PERTURB" ] || [ "$DEFAULT_OGBENCH_RESET_PERTURB" = "null" ] || [ "$DEFAULT_OGBENCH_RESET_PERTURB" = "None" ]; then
    DEFAULT_OGBENCH_RESET_PERTURB="true"
fi

echo ""
echo "Enable OGBench start/goal perturbation for both waypoint search and benchmark rollout? [default: ${DEFAULT_OGBENCH_RESET_PERTURB}]"
while true; do
    read -r -p "Enable perturbation (true/false): " _perturb_input
    _perturb_input="${_perturb_input:-$DEFAULT_OGBENCH_RESET_PERTURB}"
    case "${_perturb_input,,}" in
        true|t|yes|y|1|on)
            OGBENCH_ENABLE_RESET_PERTURB="true"
            break
            ;;
        false|f|no|n|0|off)
            OGBENCH_ENABLE_RESET_PERTURB="false"
            break
            ;;
    esac
    echo "Please enter true or false."
done

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
GRAPH_SAMPLE_RATIO="$(read_yaml_value "$PLANNING_CONFIG_SRC" "sampled_graph_sample_ratio")"
GRAPH_EDGE_RADIUS="$(read_yaml_value "$PLANNING_CONFIG_SRC" "sampled_graph_edge_radius")"
GRAPH_SEED="$(read_yaml_value "$PLANNING_CONFIG_SRC" "sampled_graph_seed")"
GRAPH_CACHE_HOST_DIR="$(resolve_graph_cache_host_dir "$(read_yaml_value "$PLANNING_CONFIG_SRC" "sampled_graph_save_dir")")"
TASK_OVERRIDE_HOST_PATH="$(resolve_repo_path "$TASK_OVERRIDE_PATH")"
EXISTING_CACHE_RAW_PATH=""
EXISTING_CACHE_HOST_PATH=""
EXISTING_RESET_PERTURB=""
if [ -f "$TASK_OVERRIDE_HOST_PATH" ]; then
    EXISTING_CACHE_RAW_PATH="$(read_yaml_value "$TASK_OVERRIDE_HOST_PATH" "sampled_graph_cache_path")"
    EXISTING_RESET_PERTURB="$(read_yaml_value "$TASK_OVERRIDE_HOST_PATH" "ogbench_enable_reset_perturb")"
    if [ -n "$EXISTING_CACHE_RAW_PATH" ] && [ "$EXISTING_CACHE_RAW_PATH" != "None" ] && [ "$EXISTING_CACHE_RAW_PATH" != "null" ]; then
        EXISTING_CACHE_HOST_PATH="$(resolve_graph_cache_host_path "$EXISTING_CACHE_RAW_PATH")"
    fi
fi
LATEST_MATCHING_CACHE_HOST_PATH="$(find_latest_matching_graph_cache "$GRAPH_CACHE_HOST_DIR" "$SELECTED_DATASET" "$GRAPH_SAMPLE_RATIO" "$GRAPH_EDGE_RADIUS" "$GRAPH_SEED")"
LATEST_MATCHING_CACHE_CONTAINER_PATH=""
if [ -n "$LATEST_MATCHING_CACHE_HOST_PATH" ]; then
    LATEST_MATCHING_CACHE_CONTAINER_PATH="$(to_graph_cache_container_path "$LATEST_MATCHING_CACHE_HOST_PATH")"
fi

REGENERATE_TASK_OVERRIDE=0
if [ ! -f "$TASK_OVERRIDE_HOST_PATH" ] || [ -z "$EXISTING_CACHE_HOST_PATH" ] || [ ! -f "$EXISTING_CACHE_HOST_PATH" ]; then
    REGENERATE_TASK_OVERRIDE=1
elif [ "$EXISTING_RESET_PERTURB" != "$OGBENCH_ENABLE_RESET_PERTURB" ]; then
    REGENERATE_TASK_OVERRIDE=1
elif [ -z "$LATEST_MATCHING_CACHE_HOST_PATH" ]; then
    REGENERATE_TASK_OVERRIDE=1
elif [ "$EXISTING_CACHE_HOST_PATH" != "$LATEST_MATCHING_CACHE_HOST_PATH" ]; then
    REGENERATE_TASK_OVERRIDE=1
elif [ "$EXISTING_CACHE_RAW_PATH" != "$LATEST_MATCHING_CACHE_CONTAINER_PATH" ]; then
    REGENERATE_TASK_OVERRIDE=1
fi

if [ "$REGENERATE_TASK_OVERRIDE" -eq 1 ]; then
    if [ -n "$LATEST_MATCHING_CACHE_CONTAINER_PATH" ]; then
        GRAPH_CACHE_PATH_FOR_SCRIPT="$LATEST_MATCHING_CACHE_CONTAINER_PATH"
        GRAPH_CACHE_HOST_PATH="$LATEST_MATCHING_CACHE_HOST_PATH"
    else
        GRAPH_CACHE_HOST_PATH="$(python3 - "$GRAPH_CACHE_HOST_DIR" "$SELECTED_DATASET" "$GRAPH_SAMPLE_RATIO" "$GRAPH_EDGE_RADIUS" "$GRAPH_SEED" "$RUN_TIMESTAMP" <<'PY'
import sys
import os

cache_dir, dataset, sample_ratio, edge_radius, seed, timestamp = sys.argv[1:]

def _float_token(value: str) -> str:
    return f"{float(value):g}".replace("-", "m")

print(
    os.path.join(
    cache_dir,
    f"{dataset}_sampled_graph_"
    f"r{_float_token(sample_ratio)}_"
    f"seed{int(seed)}_"
    f"rad{_float_token(edge_radius)}_"
    f"ts{timestamp}.pkl")
)
PY
)"
        GRAPH_CACHE_PATH_FOR_SCRIPT="$(to_graph_cache_container_path "$GRAPH_CACHE_HOST_PATH")"
    fi
    echo ""
    echo "Preparing waypoint override via metric-disagreement search..."
    echo "  Feasible points : $FEASIBLE_PATH"
    echo "  Override output : $TASK_OVERRIDE_PATH"
    echo "  Graph cache     : $GRAPH_CACHE_PATH_FOR_SCRIPT"
    echo "  Perturb enabled : $OGBENCH_ENABLE_RESET_PERTURB"

    DOCKER_EXTRA_ARGS=()
    if [ -d /usr/lib/wsl ]; then
        DOCKER_EXTRA_ARGS+=(-v /usr/lib/wsl:/usr/lib/wsl)
    fi

    docker run --rm \
      -e MUJOCO_GL=osmesa \
      -e HYDRA_FULL_ERROR=1 \
      -e LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/${DOCKER_USER}/.mujoco/mujoco210/bin" \
      -v "${PROJECT_DIR}:${DOCKER_PROJECT}" \
      -v "${PROJECT_DIR}/../ogbench_data:/home/${DOCKER_USER}/ogbench_data" \
      -v "$(dirname "${PROJECT_DIR}")/ogbench_data:/home/${DOCKER_USER}/.ogbench/data" \
      "${DOCKER_EXTRA_ARGS[@]}" \
      -w "${DOCKER_PROJECT}" \
      "${DOCKER_IMAGE}" \
      python3 scripts/find_waypoint_metric_disagreements.py \
        --ckpt "${DOCKER_PROJECT}/outputs/downloaded/${WANDB_ENTITY}/${WANDB_PROJECT}/${SELECTED_MODEL_ID}/model.ckpt" \
        --feasible-points-path "${DOCKER_PROJECT}/${FEASIBLE_PATH}" \
        --out "${DOCKER_PROJECT}/${TASK_OVERRIDE_PATH}" \
        --graph-cache-path "${GRAPH_CACHE_PATH_FOR_SCRIPT}" \
        --ogbench-enable-reset-perturb "${OGBENCH_ENABLE_RESET_PERTURB}" \
        --num-waypoints 3

    TASK_OVERRIDE_HOST_PATH="$(resolve_repo_path "$TASK_OVERRIDE_PATH")"
    if [ ! -f "$TASK_OVERRIDE_HOST_PATH" ]; then
        echo "ERROR: Failed to generate task override file: $TASK_OVERRIDE_HOST_PATH"
        exit 1
    fi
fi

RESULTS_FILE_PREFIX="${SELECTED_DATASET}_hamiltonian_${RUN_TIMESTAMP}"
RESULTS_RUN_DIR="benchmark_results/${SELECTED_MODEL_ID}/${RESULTS_FILE_PREFIX}"
SUMMARY_JSON_REL="${RESULTS_RUN_DIR}/${RESULTS_FILE_PREFIX}_summary.json"
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
echo "  Perturb enabled     : $OGBENCH_ENABLE_RESET_PERTURB"
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
    --results_file_prefix "$RESULTS_FILE_PREFIX" \
    --ogbench-enable-reset-perturb "$OGBENCH_ENABLE_RESET_PERTURB"

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
