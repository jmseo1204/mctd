#!/usr/bin/env bash
# run_kde_grad_test.sh
# Docker로 test_kde_guidance_grad.py 실행
# Usage:
#   bash run_kde_grad_test.sh                          # 기본값
#   bash run_kde_grad_test.sh --sigma 2.0 --lam 3.0 --sample_ratio 0.5  # 파라미터 조정
#   bash run_kde_grad_test.sh --goal 40 8 --out debug/my_test.png

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/project_config.sh
source "$PROJECT_DIR/scripts/project_config.sh"

# 데이터셋: ~/.ogbench/data 우선, 없으면 ../ogbench_data fallback
OGBENCH_DATA_DIR="/home/${DOCKER_USER}/.ogbench/data"
if [ -d "$HOME/.ogbench/data" ]; then
    OGBENCH_HOST_DATA="$HOME/.ogbench/data"
else
    OGBENCH_HOST_DATA="$(realpath "$PROJECT_DIR/../ogbench_data")"
fi

# HILP: run_jobs.py 와 동일한 ../HILP 경로
HILP_HOST_DIR="$(realpath "$PROJECT_DIR/../HILP")"
HILP_DOCKER_DIR="/home/${DOCKER_USER}/HILP"

echo "==================================================="
echo "  KDE Guidance Gradient Visualizer"
echo "==================================================="
echo "  project: $PROJECT_DIR"
echo "  data:    $OGBENCH_HOST_DATA"
echo "  HILP:    $HILP_HOST_DIR"
echo "  args:    $*"
echo "==================================================="

docker run --rm --gpus all \
    -e MUJOCO_GL=osmesa \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/${DOCKER_USER}/.mujoco/mujoco210/bin \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -v "${PROJECT_DIR}":/home/${DOCKER_USER}/mctd \
    -v "${OGBENCH_HOST_DATA}":"${OGBENCH_DATA_DIR}" \
    -v "${HILP_HOST_DIR}":"${HILP_DOCKER_DIR}" \
    "${DOCKER_IMAGE}" \
    /bin/bash -c "
        cd /home/${DOCKER_USER}/mctd
        python3 debug/test_kde_guidance_grad.py $*
    "

echo ""
echo "Done. Output: ${PROJECT_DIR}/debug/kde_grad_viz.png"
echo "(--out 옵션으로 경로 변경 가능)"
