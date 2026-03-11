#!/bin/bash
# ====================================================
# MCTD Docker Image Build Script
# ====================================================
# Usage: bash dockerfile/build.sh
#   (Run from the project root: mctd_repo/)
# ====================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
PLUGIN_DIR="/usr/local/lib/docker/cli-plugins"
SOURCE_DIR="/usr/libexec/docker/cli-plugins"

echo "===================================================="
echo "  MCTD Docker Image Build"
echo "===================================================="
echo ""

# ─────────────────────────────────────────────────────
# 1. Check Docker daemon
# ─────────────────────────────────────────────────────
echo "[1] Checking Docker daemon..."
if ! docker ps > /dev/null 2>&1; then
    echo "  Docker daemon not running. Starting..."
    sudo service docker start
    sleep 2
fi
echo "  ✓ Docker daemon running"
echo ""

# ─────────────────────────────────────────────────────
# 2. Fix buildx plugin symlink
#    (Removes broken Docker Desktop symlinks and creates
#     fresh ones pointing to the installed docker-ce plugins)
# ─────────────────────────────────────────────────────
echo "[2] Fixing Docker CLI plugin symlinks..."
sudo mkdir -p "$PLUGIN_DIR"

# Remove broken symlinks (leftover from Docker Desktop)
for f in "$PLUGIN_DIR"/*; do
    if [ -L "$f" ] && [ ! -e "$f" ]; then
        echo "  removing broken symlink: $(basename $f)"
        sudo rm "$f"
    fi
done

# Create buildx symlink
if [ -f "$SOURCE_DIR/docker-buildx" ]; then
    sudo ln -sf "$SOURCE_DIR/docker-buildx" "$PLUGIN_DIR/docker-buildx"
    echo "  ✓ docker-buildx symlink ready"
else
    echo "  ✗ docker-buildx not found at $SOURCE_DIR"
    echo "    Install with: sudo apt-get install docker-buildx-plugin"
    exit 1
fi

# Create compose symlink
if [ -f "$SOURCE_DIR/docker-compose" ]; then
    sudo ln -sf "$SOURCE_DIR/docker-compose" "$PLUGIN_DIR/docker-compose"
    echo "  ✓ docker-compose symlink ready"
fi
echo ""

# ─────────────────────────────────────────────────────
# 3. Build Docker image
# ─────────────────────────────────────────────────────
echo "[3] Building Docker image: $DOCKER_IMAGE"
echo "  Dockerfile: dockerfile/Dockerfile"
echo "  Context:    $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"
docker build -f dockerfile/Dockerfile -t "$DOCKER_IMAGE" .

echo ""
echo "===================================================="
echo "  ✓ Build complete: $DOCKER_IMAGE"
echo "  Verify with: docker images | grep mctd"
echo "===================================================="
