#!/usr/bin/env bash
# scripts/visualize_kde.sh
# Interactive KDE heatmap visualizer — runs inside mctd:0.1 Docker (same as train.sh).
#
# Usage:
#   bash scripts/visualize_kde.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Load central config (DOCKER_USER, DOCKER_IMAGE, etc.) ────────────────────
source "${SCRIPT_DIR}/project_config.sh"

DOCKER_PROJECT="/home/${DOCKER_USER}/mctd"
OGBENCH_DATA_DIR="$(dirname "${PROJECT_DIR}")/ogbench_data"
HOME_DIR="$HOME"

# ── Discover datasets from configurations/dataset/*.yaml ─────────────────────
mapfile -t DATASET_NAMES < <(
  grep -h '^dataset:' "${PROJECT_DIR}/configurations/dataset/"*.yaml 2>/dev/null \
    | awk '{gsub(/"/, "", $2); print $2}' \
    | sort -u
)

if [[ ${#DATASET_NAMES[@]} -eq 0 ]]; then
  echo "[ERROR] No dataset yaml files found under configurations/dataset/"
  exit 1
fi

# ── Interactive menu ──────────────────────────────────────────────────────────
# kde_save_dir in the yaml is the Docker-internal path (~/.ogbench/data).
# On the host that corresponds to $OGBENCH_DATA_DIR.
echo ""
echo "=========================================="
echo "  KDE Heatmap Visualizer  (Docker: ${DOCKER_IMAGE})"
echo "  Host data dir: ${OGBENCH_DATA_DIR}"
echo "=========================================="
echo ""
echo "Available datasets:"
for i in "${!DATASET_NAMES[@]}"; do
  name="${DATASET_NAMES[$i]}"
  npz_path="${OGBENCH_DATA_DIR}/${name}.npz"
  if [[ -f "${npz_path}" ]]; then
    status="✓ npz found"
  else
    status="✗ npz missing"
  fi
  printf "  [%d] %-45s  %s\n" "$((i+1))" "${name}" "${status}"
done
echo ""
read -rp "Select dataset [1-${#DATASET_NAMES[@]}]: " CHOICE

if ! [[ "${CHOICE}" =~ ^[0-9]+$ ]] || \
   (( CHOICE < 1 )) || (( CHOICE > ${#DATASET_NAMES[@]} )); then
  echo "[ERROR] Invalid selection: ${CHOICE}"
  exit 1
fi

SELECTED="${DATASET_NAMES[$((CHOICE-1))]}"
NPZ_HOST="${OGBENCH_DATA_DIR}/${SELECTED}.npz"

if [[ ! -f "${NPZ_HOST}" ]]; then
  echo "[ERROR] npz not found: ${NPZ_HOST}"
  exit 1
fi

echo ""
echo "Selected: ${SELECTED}"
echo "Running inside Docker ..."
echo ""

# ── Run kde_heatmap.py inside Docker ─────────────────────────────────────────
# Output png is written to ~/.ogbench/data inside the container
# = $OGBENCH_DATA_DIR on the host.
INNER_CMD="cd ${DOCKER_PROJECT} && \
  python3 scripts/kde_heatmap.py \
    --dataset ${SELECTED} \
    --no_show"

docker run --rm \
  -e MUJOCO_GL=osmesa \
  -e HYDRA_FULL_ERROR=1 \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/${DOCKER_USER}/.mujoco/mujoco210/bin \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -v "${PROJECT_DIR}:${DOCKER_PROJECT}" \
  -v "${OGBENCH_DATA_DIR}:/home/${DOCKER_USER}/.ogbench/data" \
  "${DOCKER_IMAGE}" /bin/bash -c "${INNER_CMD}"

# ── Open result on host ───────────────────────────────────────────────────────
OUT_PNG="${OGBENCH_DATA_DIR}/${SELECTED}_kde_heatmap.png"
if [[ -f "${OUT_PNG}" ]]; then
  echo ""
  echo "Saved → ${OUT_PNG}"
  # WSL: open with Windows Photo Viewer via explorer.exe (avoids xdg-open → Foxglove)
  if command -v explorer.exe &>/dev/null; then
    explorer.exe "$(wslpath -w "${OUT_PNG}")" &
  elif command -v eog &>/dev/null; then
    eog "${OUT_PNG}" &
  elif command -v display &>/dev/null; then
    display "${OUT_PNG}" &
  else
    echo "(No image viewer found — open the file manually.)"
  fi
fi
