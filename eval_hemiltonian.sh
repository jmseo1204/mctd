#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DATASET_DIR="$PROJECT_DIR/configurations/dataset"
PLANNING_CONFIG_SRC="$PROJECT_DIR/configurations/algorithm/df_planning.yaml"

MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

OUTPUT_DOWNLOADED_DIR="$MCTD_DOWNLOADED_DIR"
NUM_TASKS=5
NUM_SEEDS=1
START_TASK_IDX=1

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

ENV_ID="$(read_yaml_value "$DATASET_YAML" "env_id")"
if [ -z "$ENV_ID" ]; then
    echo "ERROR: Could not determine env_id from ${DATASET_YAML}"
    exit 1
fi
ENV_KEY="$(normalize_env_key "$ENV_ID")"
CONFIG_TASK_OVERRIDE_PATH="$(read_yaml_value "$PLANNING_CONFIG_SRC" "task_override_path")"
DEFAULT_OVERRIDE_PATH="configurations/task_overrides/${ENV_KEY}_waypoints.yaml"
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
if [ ! -f "${PROJECT_DIR}/${TASK_OVERRIDE_PATH}" ]; then
    echo "ERROR: Task override file not found: ${PROJECT_DIR}/${TASK_OVERRIDE_PATH}"
    echo "       Run eval_hemiltonian_all.sh once to generate ${DEFAULT_OVERRIDE_PATH} if needed."
    exit 1
fi
TASK_OVERRIDE_CACHE_PATH="$(python3 - "${PROJECT_DIR}/${TASK_OVERRIDE_PATH}" "${PROJECT_DIR}" <<'PY'
import sys
import yaml

override_path, _repo_root = sys.argv[1], sys.argv[2]
with open(override_path, "r", encoding="utf-8") as f:
    payload = yaml.safe_load(f) or {}
cache_path = payload.get("sampled_graph_cache_path")
if cache_path in (None, ""):
    print("")
    raise SystemExit(0)
print(str(cache_path))
PY
)"
TASK_OVERRIDE_CACHE_PATH="$(resolve_graph_cache_host_path "$TASK_OVERRIDE_CACHE_PATH")"
if [ -z "$TASK_OVERRIDE_CACHE_PATH" ] || [ ! -f "$TASK_OVERRIDE_CACHE_PATH" ]; then
    echo "ERROR: Task override file is stale or missing sampled_graph_cache_path: ${PROJECT_DIR}/${TASK_OVERRIDE_PATH}"
    echo "       Regenerate it via eval_hemiltonian_all.sh so the waypoint override is pinned to a concrete graph cache."
    exit 1
fi
WAYPOINT_GROUP_IDX=""
if [ "$TASK_OVERRIDE_PATH" != "none" ]; then
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
