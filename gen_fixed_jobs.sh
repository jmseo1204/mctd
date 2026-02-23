#!/bin/bash

# ====================================================
# MCTD Generalized Evaluation Launcher (Fixed Inputs)
# ====================================================
# Fixed inputs: 3-10-3-1
# - Dataset index: 3 (og_antmaze_giant_navigate)
# - Model index: 10
# - Num tasks: 1
# - Num seeds: 1

PROJECT_DIR=$(pwd)
CONFIG_DATASET_DIR="configurations/dataset"
OUTPUT_DOWNLOADED_DIR="outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval"

echo "===================================================="
echo "  MCTD Job Generator (Fixed: 3-10-3-1)"
echo "===================================================="

# 0. Clean up all existing MCTD containers and jobs
echo "Cleaning up existing MCTD containers..."
docker rm -f $(docker ps -a 2>/dev/null | grep exp_gpu0 | awk '{print $1}') 2>/dev/null || true
echo "✓ Container cleanup complete"

echo "Cleaning up existing job files..."
rm -f jobs/*.json
echo "✓ Job files cleanup complete"
echo ""

# 1. Dataset Selection (Fixed: index 3)
echo "Searching for datasets in: ${CONFIG_DATASET_DIR}"
datasets=( $(ls ${CONFIG_DATASET_DIR}/*.yaml | xargs -n 1 basename | sed 's/\.yaml//' | grep -v "base") )

if [ ${#datasets[@]} -eq 0 ]; then
    echo "No datasets found in ${CONFIG_DATASET_DIR}"
    exit 1
fi

ds_idx=3
SELECTED_DATASET="${datasets[$ds_idx]}"
echo "✓ Selected dataset (index $ds_idx): $SELECTED_DATASET"

if [ -z "$SELECTED_DATASET" ]; then
    echo "Invalid dataset selection."
    exit 1
fi

# 2. Model Selection (Fixed: index 10)
echo ""
echo "Searching for downloaded models in: ${OUTPUT_DOWNLOADED_DIR}"
if [ ! -d "$OUTPUT_DOWNLOADED_DIR" ]; then
    echo "Directory ${OUTPUT_DOWNLOADED_DIR} not found. Trying to list all model.ckpt files in outputs..."
    ckpt_files=( $(find outputs/downloaded -name "model.ckpt") )
else
    ckpt_files=( $(find "$OUTPUT_DOWNLOADED_DIR" -name "model.ckpt") )
fi

if [ ${#ckpt_files[@]} -eq 0 ]; then
    echo "No model.ckpt files found."
    exit 1
fi

# Get model IDs
model_ids=()
for ckpt in "${ckpt_files[@]}"; do
    model_id=$(basename $(dirname "$ckpt"))
    model_ids+=("$model_id")
done

mod_idx=10
SELECTED_MODEL_ID="${model_ids[$mod_idx]}"
echo "✓ Selected model (index $mod_idx): $SELECTED_MODEL_ID"

if [ -z "$SELECTED_MODEL_ID" ]; then
    echo "Invalid model selection."
    exit 1
fi

# 3. Fixed Parameters
NUM_TASKS=1
NUM_SEEDS=1
NUM_REPEATS=1

echo "✓ Num tasks: $NUM_TASKS"
echo "✓ Num seeds: $NUM_SEEDS"
echo "✓ Num repeats: $NUM_REPEATS"

# 4. Execute Job Generator
echo ""
echo "Running: python3 generate_jobs_generalized.py --dataset $SELECTED_DATASET --model_id $SELECTED_MODEL_ID --num_tasks $NUM_TASKS --num_seeds $NUM_SEEDS --num_repeats $NUM_REPEATS"
python3 generate_jobs_generalized.py \
    --dataset "$SELECTED_DATASET" \
    --model_id "$SELECTED_MODEL_ID" \
    --num_tasks "$NUM_TASKS" \
    --num_seeds "$NUM_SEEDS" \
    --num_repeats "$NUM_REPEATS"

echo ""
echo "Generation Complete."
echo "You can now run the jobs using: python3 run_jobs.py"
