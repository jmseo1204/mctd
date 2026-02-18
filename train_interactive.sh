#!/bin/bash

# ====================================================
# MCTD Master Interactive Training Script
# ====================================================
# This script replaces all previous training scripts.
# Features: Dataset selection, Window size adjustment, 
#           Auto-resume detection, Epoch-level progress bar.

# 1. Path Configuration
OGBENCH_DATA_DIR="/mnt/c/Users/USER/Desktop/test_ogbench/ogbench_data"
CONFIG_DATASET_DIR="configurations/dataset"
PROJECT_DIR=$(pwd)
OUTPUT_MOUNT_DIR="/home/jmseo1204/mctd_outputs"
HOME_DIR=$HOME
DOCKER_IMAGE="fmctd:0.1"
DOCKER_USER="jsyoon"
ENTITY="jmseo1204-seoul-national-university"
PROJECT="mctd_eval"

# 2. Dataset Selection
echo "===================================================="
echo "      MCTD Master Training Launcher"
echo "===================================================="
echo "Searching for datasets in: ${OGBENCH_DATA_DIR}"

datasets=( $(ls ${OGBENCH_DATA_DIR}/*.npz 2>/dev/null | xargs -n 1 basename | sed 's/\.npz//') )
for i in "${!datasets[@]}"; do
    printf "  [%2d] %s\n" "$i" "${datasets[$i]}"
done
echo "----------------------------------------------------"
read -p "Select dataset index: " dataset_idx
SELECTED_DATASET="${datasets[$dataset_idx]}"
[ -z "$SELECTED_DATASET" ] && { echo "Invalid selection"; exit 1; }

# 3. Resume Check
echo "Checking for previous runs of ${SELECTED_DATASET} to resume..."
RESUME_ID=$(python3 - <<EOF
import os
from pathlib import Path
import yaml

output_base = Path("outputs")
target_dataset = "${SELECTED_DATASET}"
download_root = Path("outputs/downloaded") / "${ENTITY}" / "${PROJECT}"

# 1. Check outputs/downloaded first for valid resumable checkpoints
if download_root.exists():
    # Sort by modification time of the model.ckpt file
    ckpts = []
    for d in download_root.iterdir():
        ckpt_file = d / "model.ckpt"
        if ckpt_file.exists():
            ckpts.append((d.name, ckpt_file.stat().st_mtime))
    
    if ckpts:
        # Sort and get the latest
        ckpts.sort(key=lambda x: x[1], reverse=True)
        print(ckpts[0][0])
        exit(0)
EOF
)

RESUME_ARG=""
if [ ! -z "$RESUME_ID" ]; then
    read -p ">> Previous run found ($RESUME_ID). Resume training from this checkpoint? [Y/n]: " DO_RESUME
    DO_RESUME=${DO_RESUME:-y}
    if [[ "$DO_RESUME" == "y" || "$DO_RESUME" == "Y" ]]; then
        RESUME_ARG="resume=$RESUME_ID"
        echo "   [Mode] RESUME active. Progress will be preserved."
    fi
fi

# 4. Parameter Input
if [ -z "$RESUME_ARG" ]; then
    read -p "Enter training episode length (frame_stack: 10, recommended: 50, 100): " USER_EP_LEN
    [ $((USER_EP_LEN % 10)) -ne 0 ] && { echo "Must be divisible by 10"; exit 1; }
else
    # If resuming, we should ideally use the same episode length. 
    # Let's extract it from the config if possible, or ask.
    echo ">> Resuming... episode_len will be loaded from checkpoint."
fi

# 5. Configuration Matching
mkdir -p ${OUTPUT_MOUNT_DIR}
chmod 777 ${OUTPUT_MOUNT_DIR}
YAML_NAME="og_${SELECTED_DATASET//-/_}"
if [ ! -f "${CONFIG_DATASET_DIR}/${YAML_NAME}.yaml" ]; then
    DATASET_CONFIG="og_antmaze_giant_stitch" # Fallback
else
    DATASET_CONFIG="$YAML_NAME"
fi

# 6. Docker Execution
docker rm -f train_interactive 2>/dev/null
echo "Launching Docker container... (Epoch-level progress bar is enabled)"

# Prepare optional arguments
OPTS=""
[ ! -z "$RESUME_ARG" ] && OPTS="$OPTS $RESUME_ARG"
[ ! -z "$USER_EP_LEN" ] && OPTS="$OPTS dataset.episode_len=$USER_EP_LEN algorithm.chunk_size=$USER_EP_LEN"

docker run --rm -it --gpus all --name train_interactive --shm-size=50g \
    -e MUJOCO_GL=osmesa \
    -v ${PROJECT_DIR}:/home/${DOCKER_USER}/mctd \
    -v ${OUTPUT_MOUNT_DIR}:/home/${DOCKER_USER}/mctd/outputs \
    -v ${OGBENCH_DATA_DIR}:/home/${DOCKER_USER}/.ogbench/data \
    -v ${HOME_DIR}/.netrc:/home/${DOCKER_USER}/.netrc \
    -v ${HOME_DIR}/.d4rl:/home/${DOCKER_USER}/.d4rl \
    -v ${HOME_DIR}/.ogbench:/home/${DOCKER_USER}/.ogbench \
    ${DOCKER_IMAGE} /bin/bash \
    -c "cd mctd && python3 main.py experiment.tasks=[training] \
        experiment=exp_planning \
        algorithm=df_planning \
        dataset=${DATASET_CONFIG} \
        dataset.dataset=${SELECTED_DATASET} \
        dataset.save_dir=/home/jsyoon/.ogbench/data \
        +name=Interactive_${SELECTED_DATASET} \
        wandb.mode=online \
        experiment.training.batch_size=1024 \
        $OPTS"

# 7. Post-training: Archive model
echo "Archiving final model..."
python3 - <<EOF
import os, shutil
from pathlib import Path
output_base = Path("outputs") 
download_root = Path("outputs/downloaded") / "${ENTITY}" / "${PROJECT}"
date_dirs = sorted([d for d in output_base.glob("202*") if d.is_dir()], reverse=True)
latest_run_dir = None
for date_dir in date_dirs:
    run_dirs = sorted([d for d in date_dir.glob("*") if d.is_dir() and (d / "checkpoints").exists()], reverse=True)
    if run_dirs:
        latest_run_dir = run_dirs[0]
        break
if latest_run_dir:
    ckpt_path = latest_run_dir / "checkpoints" / "last.ckpt"
    if not ckpt_path.exists():
        ckpts = list((latest_run_dir / "checkpoints").glob("*.ckpt"))
        if ckpts: ckpt_path = ckpts[0]
    if ckpt_path.exists():
        run_id = None
        wandb_dir = latest_run_dir / "wandb"
        if wandb_dir.exists():
            run_folders = list(wandb_dir.glob("run-*"))
            if run_folders: run_id = run_folders[0].name.split("-")[-1]
        if run_id:
            target_dir = download_root / run_id
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(ckpt_path, target_dir / "model.ckpt")
            print(f"[Success] Archived: {run_id}")
EOF
