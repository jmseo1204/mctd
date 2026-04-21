#!/usr/bin/env bash
# Docker wrapper for the waypoint Hamiltonian-gap data analysis.
#
# Usage:
#   bash scripts/run_waypoint_gap_analysis.sh <SELECTED_MODEL_ID> [extra args...]
#
# Extra args are forwarded to scripts/analyze_waypoint_disagreement_gap.py.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=mctd_ckpt_lib.sh
MCTD_PROJECT_DIR="$PROJECT_DIR"
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_waypoint_gap_analysis.sh <SELECTED_MODEL_ID> [extra args...]" >&2
    exit 1
fi

SELECTED_MODEL_ID="$1"
shift || true

DOCKER_PROJECT="/home/${DOCKER_USER}/mctd"
CKPT_PATH_HOST="${MCTD_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/model.ckpt"
if [ ! -f "$CKPT_PATH_HOST" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH_HOST" >&2
    exit 1
fi
CKPT_PATH_CTR="${DOCKER_PROJECT}/outputs/downloaded/${WANDB_ENTITY}/${WANDB_PROJECT}/${SELECTED_MODEL_ID}/model.ckpt"

DOCKER_EXTRA_ARGS=()
if [ -d /usr/lib/wsl ]; then
    DOCKER_EXTRA_ARGS+=(-v /usr/lib/wsl:/usr/lib/wsl)
fi

echo "Running waypoint gap feature extraction in Docker (image: ${DOCKER_IMAGE})..."
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
  python3 scripts/analyze_waypoint_disagreement_gap.py \
    --ckpt "${CKPT_PATH_CTR}" \
    "$@"

echo ""
echo "Generating report (Docker)..."
docker run --rm \
  -v "${PROJECT_DIR}:${DOCKER_PROJECT}" \
  "${DOCKER_EXTRA_ARGS[@]}" \
  -w "${DOCKER_PROJECT}" \
  "${DOCKER_IMAGE}" \
  python3 scripts/analyze_waypoint_disagreement_report.py

echo ""
echo "Done. See outputs/waypoint_gap_analysis/report.md"
