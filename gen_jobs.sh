#!/bin/bash

# ====================================================
# MCTD Generalized Evaluation Launcher
# ====================================================

PROJECT_DIR=$(pwd)
CONFIG_DATASET_DIR="configurations/dataset"
# symlink path inside repo
OUTPUT_DOWNLOADED_DIR="outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval"

echo "===================================================="
echo "      MCTD Generalized Evaluation Launcher"
echo "===================================================="

# 1. Dataset Selection
echo "Search for datasets in: ${CONFIG_DATASET_DIR}"
datasets=( $(ls ${CONFIG_DATASET_DIR}/*.yaml | xargs -n 1 basename | sed 's/\.yaml//' | grep -v "base") )

if [ ${#datasets[@]} -eq 0 ]; then
    echo "No datasets found in ${CONFIG_DATASET_DIR}"
    exit 1
fi

for i in "${!datasets[@]}"; do
    printf "  [%2d] %s\n" "$i" "${datasets[$i]}"
done
read -p "Select dataset index: " ds_idx
SELECTED_DATASET="${datasets[$ds_idx]}"

if [ -z "$SELECTED_DATASET" ]; then
    echo "Invalid dataset selection."
    exit 1
fi

# 2. Model Selection
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
    read -p "Manual Model ID input: " SELECTED_MODEL_ID
else
    # Show IDs (the parent directory name)
    model_ids=()
    for ckpt in "${ckpt_files[@]}"; do
        model_id=$(basename $(dirname "$ckpt"))
        model_ids+=("$model_id")
    done
    
    for i in "${!model_ids[@]}"; do
        printf "  [%2d] %s\n" "$i" "${model_ids[$i]}"
    done
    read -p "Select model index: " mod_idx
    SELECTED_MODEL_ID="${model_ids[$mod_idx]}"
fi

if [ -z "$SELECTED_MODEL_ID" ]; then
    echo "Invalid model selection."
    exit 1
fi

# 3. Environment/Task specific suggestions
MAX_TASKS=5
if [[ "$SELECTED_DATASET" == *"giant"* ]]; then
    MAX_TASKS=5
    echo "Notice: Giant Maze typically has 5 tasks."
elif [[ "$SELECTED_DATASET" == *"large"* ]]; then
    MAX_TASKS=10
    echo "Notice: Large Maze typically has 10 tasks."
fi

# 4. Interactive Inputs
echo ""
read -p "Enter number of tasks (Max recommended $MAX_TASKS): " NUM_TASKS
read -p "Enter number of seeds per task (e.g., 3): " NUM_SEEDS

NUM_TASKS=${NUM_TASKS:-$MAX_TASKS}
NUM_SEEDS=${NUM_SEEDS:-3}
NUM_REPEATS=1

# 5. Execute Job Generator
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
