#!/usr/bin/env bash
# Interactive temporal-distance heatmap visualizer.
#
# Produces one PNG with:
#   1) temporal-distance heatmap + existing grad field
#   2) sampled-graph goal distances + snapped waypoint Hamiltonian route

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

echo "============================================"
echo "  Temporal Distance Visualizer"
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

echo ""
echo "[Step 2] Reference task"
read -rp "Task ID [default: 1]: " TASK_ID
TASK_ID="${TASK_ID:-1}"
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || [ "${TASK_ID}" -lt 1 ]; then
  echo "[ERROR] Invalid task id: ${TASK_ID}"
  exit 1
fi

echo ""
echo "[Step 3] Task override / waypoints"
DEFAULT_OVERRIDE_PATH="configurations/task_overrides/antmaze_giant_waypoints.yaml"
read -rp "Task override path [Enter=${DEFAULT_OVERRIDE_PATH}, none=disable]: " TASK_OVERRIDE_PATH
TASK_OVERRIDE_PATH="${TASK_OVERRIDE_PATH:-${DEFAULT_OVERRIDE_PATH}}"

OVERRIDE_ARGS=()
OUTPUT_TAG="task_goal"
WAYPOINT_GROUP_IDX=""
if [ "${TASK_OVERRIDE_PATH}" != "none" ]; then
  if [[ "${TASK_OVERRIDE_PATH}" = /* ]]; then
    if [[ "${TASK_OVERRIDE_PATH}" == "${PROJECT_DIR}/"* ]]; then
      TASK_OVERRIDE_PATH="${TASK_OVERRIDE_PATH#${PROJECT_DIR}/}"
    else
      echo "[ERROR] Task override path must be inside ${PROJECT_DIR} so Docker can access it."
      exit 1
    fi
  fi
  if [ ! -f "${PROJECT_DIR}/${TASK_OVERRIDE_PATH}" ]; then
    echo "[ERROR] Task override file not found: ${PROJECT_DIR}/${TASK_OVERRIDE_PATH}"
    exit 1
  fi
  OVERRIDE_ARGS+=(--task_override_path "${TASK_OVERRIDE_PATH}")
  OUTPUT_TAG="$(basename "${TASK_OVERRIDE_PATH}")"
  OUTPUT_TAG="${OUTPUT_TAG%.*}"
  OUTPUT_TAG="${OUTPUT_TAG//[^0-9A-Za-z._-]/_}"

  read -rp "Waypoint group index [Enter=active/default]: " WAYPOINT_GROUP_IDX
  if [ -n "${WAYPOINT_GROUP_IDX}" ]; then
    if ! [[ "${WAYPOINT_GROUP_IDX}" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] Waypoint group index must be a non-negative integer."
      exit 1
    fi
    OVERRIDE_ARGS+=(--waypoint_group_idx "${WAYPOINT_GROUP_IDX}")
    OUTPUT_TAG="${OUTPUT_TAG}_g${WAYPOINT_GROUP_IDX}"
  fi
else
  TASK_OVERRIDE_PATH=""
fi

OUT_HOST="${PROJECT_DIR}/visualizations/${SELECTED_MODEL_ID}_${SELECTED_DATASET}_task${TASK_ID}_${OUTPUT_TAG}_temporal_dist_viz.png"
OUT_DOCKER="${DOCKER_PROJECT}/visualizations/${SELECTED_MODEL_ID}_${SELECTED_DATASET}_task${TASK_ID}_${OUTPUT_TAG}_temporal_dist_viz.png"

echo ""
echo "============================================"
echo "  Temporal Distance Heatmap + Waypoint Route"
echo "  model      : ${SELECTED_MODEL_ID}"
echo "  dataset    : ${SELECTED_DATASET}"
echo "  task_id    : ${TASK_ID}"
if [ ${#OVERRIDE_ARGS[@]} -eq 0 ]; then
  echo "  override   : none"
else
  echo "  override   : ${TASK_OVERRIDE_PATH}"
  if [ -n "${WAYPOINT_GROUP_IDX}" ]; then
    echo "  group_idx  : ${WAYPOINT_GROUP_IDX}"
  else
    echo "  group_idx  : active/default"
  fi
fi
echo "  path       : override start -> waypoints -> goal"
echo "  output     : ${OUT_HOST}"
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
  python3 scripts/temporal_dist_heatmap.py \
    --ckpt "${CKPT_DOCKER}" \
    --task_id "${TASK_ID}" \
    --grid_res 100 \
    --grad_grid_step 2.0 \
    --out "${OUT_DOCKER}" \
    "${OVERRIDE_ARGS[@]}" \
    --no_show

if [ -f "${OUT_HOST}" ]; then
  echo ""
  echo "Saved -> ${OUT_HOST}"
fi
