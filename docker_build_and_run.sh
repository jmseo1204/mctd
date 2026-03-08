#!/bin/bash

# ====================================================
# MCTD Docker Build and Run Script
# ====================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
DOCKER_IMAGE_NAME="mctd"
DOCKER_IMAGE_TAG="latest"
DOCKER_USER="jsyoon"
CONTAINER_NAME="mctd_container"

# Data paths (adjust as needed)
OGBENCH_DATA_DIR="/mnt/c/Users/USER/Desktop/test_ogbench/ogbench_data"
OUTPUT_DIR="${HOME}/mctd_outputs"

echo "====================================================="
echo "  MCTD Docker Build & Run Script"
echo "====================================================="
echo ""

# ─────────────────────────────────────────────────────
# 1. Build Image
# ─────────────────────────────────────────────────────
echo "[1] Building Docker image..."
echo "  Image: ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
echo "  Context: ${PROJECT_DIR}"
echo ""

if docker build -f "${PROJECT_DIR}/Dockerfile.prod" \
   -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" \
   "${PROJECT_DIR}"; then
  echo "✓ Image built successfully: ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
else
  echo "❌ Failed to build image"
  exit 1
fi
echo ""

# ─────────────────────────────────────────────────────
# 2. Prepare directories
# ─────────────────────────────────────────────────────
echo "[2] Preparing directories..."
mkdir -p "$OUTPUT_DIR"
chmod 777 "$OUTPUT_DIR"
echo "✓ Output directory ready: $OUTPUT_DIR"
echo ""

# ─────────────────────────────────────────────────────
# 3. Remove old container if exists
# ─────────────────────────────────────────────────────
echo "[3] Cleaning up old containers..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
echo "✓ Cleanup complete"
echo ""

# ─────────────────────────────────────────────────────
# 4. Run Container
# ─────────────────────────────────────────────────────
echo "[4] Starting container..."
echo "  Name: ${CONTAINER_NAME}"
echo "  Project mount: ${PROJECT_DIR} → /home/${DOCKER_USER}/mctd"
echo "  Data mount: ${OGBENCH_DATA_DIR} → /home/${DOCKER_USER}/.ogbench/data"
echo "  Output mount: ${OUTPUT_DIR} → /home/${DOCKER_USER}/mctd/outputs"
echo ""

docker run --rm -it --gpus all \
  --name "$CONTAINER_NAME" \
  --shm-size=50g \
  -e MUJOCO_GL=osmesa \
  -v "${PROJECT_DIR}:/home/${DOCKER_USER}/mctd" \
  -v "${OUTPUT_DIR}:/home/${DOCKER_USER}/mctd/outputs" \
  -v "${OGBENCH_DATA_DIR}:/home/${DOCKER_USER}/.ogbench/data" \
  "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"

echo ""
echo "====================================================="
echo "✓ Container session ended"
echo "====================================================="
