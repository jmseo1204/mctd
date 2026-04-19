#!/usr/bin/env bash
# Brute-force search for waypoint groups where temporal and graph Hamiltonian routes disagree.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/project_config.sh
source "${PROJECT_DIR}/scripts/project_config.sh"

MCTD_PROJECT_DIR="${PROJECT_DIR}"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "${PROJECT_DIR}/scripts/mctd_ckpt_lib.sh"

DOCKER_PROJECT="/home/${DOCKER_USER}/mctd"
OGBENCH_DATA_DIR="$(dirname "${PROJECT_DIR}")/ogbench_data"
OUTPUT_DOWNLOADED_DIR="${MCTD_DOWNLOADED_DIR}"

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

echo "============================================"
echo "  Waypoint Metric Disagreement Search"
echo "============================================"

if ! command -v docker &>/dev/null; then
  echo "[ERROR] Docker is not installed or not in PATH."
  exit 1
fi
if ! docker ps >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon is not running."
  exit 1
fi

echo ""
echo "[Step 1] Select checkpoint"
mctd_scan_ckpts 0
if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
  echo "[ERROR] No checkpoints found in ${MCTD_OUTPUT_MOUNT_DIR}"
  exit 1
fi
mctd_ckpt_menu --no-fresh || exit 1

SELECTED_MODEL_ID="${MCTD_SELECTED_MODEL_ID:-}"
if [ -z "${SELECTED_MODEL_ID}" ]; then
  echo "[ERROR] Could not determine selected model_id."
  exit 1
fi

mctd_ensure_eval_symlink "${MCTD_SELECTED_CKPT}" "${SELECTED_MODEL_ID}"
CKPT_HOST="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/model.ckpt"
CKPT_DOCKER="${DOCKER_PROJECT}/outputs/downloaded/${WANDB_ENTITY}/${WANDB_PROJECT}/${SELECTED_MODEL_ID}/model.ckpt"

if [ ! -f "${CKPT_HOST}" ]; then
  echo "[ERROR] Mirrored checkpoint not found: ${CKPT_HOST}"
  exit 1
fi

SELECTED_DATASET="${MCTD_SELECTED_DATASET:-unknown}"
if [ -z "${SELECTED_DATASET}" ] || [ "${SELECTED_DATASET}" = "unknown" ]; then
  TRAINING_CFG="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/training_config.yaml"
  if [ -f "${TRAINING_CFG}" ]; then
    SELECTED_DATASET="$(python3 -c "import yaml; d=yaml.safe_load(open('${TRAINING_CFG}')); print((d.get('dataset') or {}).get('config', 'unknown'))" 2>/dev/null || echo unknown)"
  fi
fi
SELECTED_DATASET="$(mctd_normalize_dataset_name "${SELECTED_DATASET}" "${PROJECT_DIR}/configurations/dataset")"
DATASET_YAML="${PROJECT_DIR}/configurations/dataset/${SELECTED_DATASET}.yaml"
if [ ! -f "${DATASET_YAML}" ]; then
  echo "[ERROR] Dataset config not found: ${DATASET_YAML}"
  exit 1
fi
ENV_ID="$(python3 -c "import yaml; d=yaml.safe_load(open('${DATASET_YAML}')); print(d.get('env_id',''))" 2>/dev/null || echo "")"
if [ -z "${ENV_ID}" ]; then
  echo "[ERROR] Could not determine env_id from ${DATASET_YAML}"
  exit 1
fi
ENV_KEY="$(normalize_env_key "${ENV_ID}")"

echo ""
echo "[Step 2] Search settings"
DEFAULT_FEASIBLE_PATH="configurations/task_overrides/${ENV_KEY}_feasible_points.yaml"
DEFAULT_OUT_PATH="configurations/task_overrides/${ENV_KEY}_waypoints.yaml"
read -rp "Feasible points path [Enter=${DEFAULT_FEASIBLE_PATH}]: " FEASIBLE_PATH
FEASIBLE_PATH="${FEASIBLE_PATH:-${DEFAULT_FEASIBLE_PATH}}"
read -rp "Output override path [Enter=${DEFAULT_OUT_PATH}]: " OUT_PATH
OUT_PATH="${OUT_PATH:-${DEFAULT_OUT_PATH}}"
read -rp "Number of waypoints N [default: 3]: " NUM_WAYPOINTS
NUM_WAYPOINTS="${NUM_WAYPOINTS:-3}"
if ! [[ "${NUM_WAYPOINTS}" =~ ^[0-9]+$ ]] || [ "${NUM_WAYPOINTS}" -lt 1 ]; then
  echo "[ERROR] Invalid number of waypoints: ${NUM_WAYPOINTS}"
  exit 1
fi
read -rp "Task IDs comma list [Enter=all tasks]: " TASK_IDS

for REL_PATH_VAR in FEASIBLE_PATH OUT_PATH; do
  REL_PATH="${!REL_PATH_VAR}"
  if [[ "${REL_PATH}" = /* ]]; then
    if [[ "${REL_PATH}" == "${PROJECT_DIR}/"* ]]; then
      REL_PATH="${REL_PATH#${PROJECT_DIR}/}"
    else
      echo "[ERROR] ${REL_PATH_VAR} must be inside ${PROJECT_DIR} so Docker can access it."
      exit 1
    fi
  fi
  printf -v "${REL_PATH_VAR}" '%s' "${REL_PATH}"
done

if [ ! -f "${PROJECT_DIR}/${FEASIBLE_PATH}" ]; then
  echo "[ERROR] Feasible points file not found: ${PROJECT_DIR}/${FEASIBLE_PATH}"
  exit 1
fi

TASK_ARGS=()
if [ -n "${TASK_IDS}" ]; then
  TASK_ARGS+=(--task-ids "${TASK_IDS}")
fi

OUT_HOST="${PROJECT_DIR}/${OUT_PATH}"
OUT_DOCKER="${DOCKER_PROJECT}/${OUT_PATH}"
FEASIBLE_DOCKER="${DOCKER_PROJECT}/${FEASIBLE_PATH}"

echo ""
echo "============================================"
echo "  Metric Disagreement Search"
echo "  model        : ${SELECTED_MODEL_ID}"
echo "  feasible     : ${FEASIBLE_PATH}"
echo "  out          : ${OUT_PATH}"
echo "  num_waypoints: ${NUM_WAYPOINTS}"
if [ -n "${TASK_IDS}" ]; then
  echo "  task_ids     : ${TASK_IDS}"
else
  echo "  task_ids     : all"
fi
echo "============================================"

DOCKER_EXTRA_ARGS=()
if [ -d /usr/lib/wsl ]; then
  DOCKER_EXTRA_ARGS+=(-v /usr/lib/wsl:/usr/lib/wsl)
fi

docker run --rm \
  -e MUJOCO_GL=osmesa \
  -e HYDRA_FULL_ERROR=1 \
  -e LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/${DOCKER_USER}/.mujoco/mujoco210/bin" \
  -v "${PROJECT_DIR}:${DOCKER_PROJECT}" \
  -v "${OGBENCH_DATA_DIR}:/home/${DOCKER_USER}/.ogbench/data" \
  "${DOCKER_EXTRA_ARGS[@]}" \
  -w "${DOCKER_PROJECT}" \
  "${DOCKER_IMAGE}" \
  python3 scripts/find_waypoint_metric_disagreements.py \
    --ckpt "${CKPT_DOCKER}" \
    --feasible-points-path "${FEASIBLE_DOCKER}" \
    --out "${OUT_DOCKER}" \
    --num-waypoints "${NUM_WAYPOINTS}" \
    "${TASK_ARGS[@]}"

if [ -f "${OUT_HOST}" ]; then
  echo ""
  echo "Saved -> ${OUT_HOST}"
fi
