#!/bin/bash

# ====================================================
# MCTD Evaluation Launcher
# ====================================================
# Prompts for state dim (2 or 29), then auto-selects
# the matching dataset config and checkpoint.

PROJECT_DIR=$(pwd)
CONFIG_DATASET_DIR="configurations/dataset"
OUTPUT_DOWNLOADED_DIR="outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval"
LOCAL_TRAIN_DIR="/home/jmseo1204/mctd_outputs"

echo "===================================================="
echo "  MCTD Job Generator"
echo "===================================================="

# Check if Docker is available
echo "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker is not installed or not in PATH"
    exit 1
fi
if ! docker ps > /dev/null 2>&1; then
    echo "❌ ERROR: Docker daemon is not running"
    exit 1
fi
echo "✓ Docker is available and running"
echo ""

# 0. Clean up existing containers and job files
echo "Cleaning up existing MCTD containers..."
docker rm -f $(docker ps -a 2>/dev/null | grep exp_gpu0 | awk '{print $1}') 2>/dev/null || true
echo "✓ Container cleanup complete"

echo "Cleaning up existing job files..."
rm -f jobs/*.json
echo "✓ Job files cleanup complete"
echo ""

# ─────────────────────────────────────────────────────
# 1. Auto-register local training checkpoints
# ─────────────────────────────────────────────────────
if [ -d "$LOCAL_TRAIN_DIR" ]; then
    echo "Scanning for local training checkpoints..."
    while IFS= read -r ckpt; do
        # Resolve symlink to actual .ckpt file
        real_ckpt=$(readlink -f "$ckpt")
        # Extract YYYY-MM-DD/HH-MM-SS from path → local_YYYYMMDD_HHMMSS
        date_time=$(echo "$real_ckpt" | grep -oP '\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}')
        if [ -z "$date_time" ]; then continue; fi
        model_id="local_$(echo "$date_time" | tr '/-' '_' | tr -d '_' | sed 's/\(....\)\(..\)\(..\)_\(..\)\(..\)\(..\)/\1\2\3_\4\5\6/')"
        dest="${OUTPUT_DOWNLOADED_DIR}/${model_id}/model.ckpt"
        if [ ! -f "$dest" ]; then
            mkdir -p "$(dirname "$dest")"
            ln -sf "$real_ckpt" "$dest"
            echo "✓ Auto-registered local checkpoint: $model_id"
        fi
    done < <(find "$LOCAL_TRAIN_DIR" -maxdepth 4 -name "last.ckpt" 2>/dev/null)
fi
echo ""

# ─────────────────────────────────────────────────────
# 2. Scan for available state dimensions
# ─────────────────────────────────────────────────────
echo "Scanning all checkpoints to identify available state dimensions..."

SCANNER_SCRIPT=$(mktemp /tmp/mctd_full_scan_XXXXXX.py)
cat > "$SCANNER_SCRIPT" <<'PYEOF'
import sys, os, torch
from pathlib import Path

base_dir = Path(sys.argv[1])
if not base_dir.exists():
    sys.exit(0)
for model_dir in sorted(base_dir.iterdir()):
    ckpt = model_dir / "model.ckpt"
    if not ckpt.exists():
        continue
    try:
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("state_dict", {})
        dm = sd.get("data_mean")
        if dm is not None:
             print(f"{model_dir.name}:{int(dm.shape[0])}")
    except:
        pass
PYEOF

SCAN_RESULTS=($(conda run -n diff_force_env python3 "$SCANNER_SCRIPT" "$OUTPUT_DOWNLOADED_DIR" 2>/dev/null))
rm -f "$SCANNER_SCRIPT"

if [ ${#SCAN_RESULTS[@]} -eq 0 ]; then
    echo "❌ ERROR: No valid checkpoints found in ${OUTPUT_DOWNLOADED_DIR}"
    exit 1
fi

# Extract unique dims
UNIQUE_DIMS=($(printf "%s\n" "${SCAN_RESULTS[@]}" | cut -d: -f2 | sort -un))

echo "Select observation state dimension:"
for i in "${!UNIQUE_DIMS[@]}"; do
    D="${UNIQUE_DIMS[$i]}"
    case "$D" in
        2)  DESC="2D  (x,y position only)   [og_antmaze_giant_navigate]" ;;
        15) DESC="15D (qpos only)            [og_antmaze_giant_navigate_15d]" ;;
        29) DESC="29D (full qpos+qvel)       [og_antmaze_giant_navigate_fullstate]" ;;
        *)  DESC="${D}D (Custom)" ;;
    esac
    echo "  $((i+1))) $DESC"
done

read -rp "Enter state dimension index [1-${#UNIQUE_DIMS[@]}]: " DIM_INDEX

if ! [[ "$DIM_INDEX" =~ ^[0-9]+$ ]] || \
   [ "$DIM_INDEX" -lt 1 ] || [ "$DIM_INDEX" -gt "${#UNIQUE_DIMS[@]}" ]; then
    echo "❌ ERROR: Invalid selection '$DIM_INDEX'"
    exit 1
fi

STATE_DIM="${UNIQUE_DIMS[$((DIM_INDEX-1))]}"
echo "✓ Selected state dim: $STATE_DIM"

case "$STATE_DIM" in
    2)  SELECTED_DATASET="og_antmaze_giant_navigate" ;;
    15) SELECTED_DATASET="og_antmaze_giant_navigate_15d" ;;
    29) SELECTED_DATASET="og_antmaze_giant_navigate_fullstate" ;;
    *)
        echo "⚠️ No mapping found for ${STATE_DIM}D. Defaulting to giant_navigate."
        SELECTED_DATASET="og_antmaze_giant_navigate"
        ;;
esac

DATASET_YAML="${CONFIG_DATASET_DIR}/${SELECTED_DATASET}.yaml"
if [ ! -f "$DATASET_YAML" ]; then
    echo "❌ ERROR: Dataset config not found: ${DATASET_YAML}"
    exit 1
fi
echo "✓ Selected dataset: $SELECTED_DATASET"
echo ""

# ─────────────────────────────────────────────────────
# 3. Filter checkpoints by selected obs_dim
# ─────────────────────────────────────────────────────
MATCHING_MODELS=()
for res in "${SCAN_RESULTS[@]}"; do
    ID=$(echo "$res" | cut -d: -f1)
    DIM=$(echo "$res" | cut -d: -f2)
    if [ "$DIM" -eq "$STATE_DIM" ]; then
        MATCHING_MODELS+=("$ID")
    fi
done

if [ ${#MATCHING_MODELS[@]} -eq 0 ]; then
    echo "❌ ERROR: No checkpoints found with obs_dim=${STATE_DIM} in ${OUTPUT_DOWNLOADED_DIR}"
    exit 1
fi

# ─────────────────────────────────────────────────────
# 4. Let user pick model (or auto-select if only one)
# ─────────────────────────────────────────────────────
if [ ${#MATCHING_MODELS[@]} -eq 1 ]; then
    SELECTED_MODEL_ID="${MATCHING_MODELS[0]}"
    echo "✓ Auto-selected model: $SELECTED_MODEL_ID (only match)"
else
    echo "Found ${#MATCHING_MODELS[@]} checkpoints with obs_dim=${STATE_DIM}:"
    for i in "${!MATCHING_MODELS[@]}"; do
        echo "  $((i+1))) ${MATCHING_MODELS[$i]}"
    done
    read -rp "Select model [1-${#MATCHING_MODELS[@]}]: " MODEL_CHOICE
    if ! [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]] || \
       [ "$MODEL_CHOICE" -lt 1 ] || [ "$MODEL_CHOICE" -gt "${#MATCHING_MODELS[@]}" ]; then
        echo "❌ ERROR: Invalid selection '$MODEL_CHOICE'"
        exit 1
    fi
    SELECTED_MODEL_ID="${MATCHING_MODELS[$((MODEL_CHOICE-1))]}"
    echo "✓ Selected model: $SELECTED_MODEL_ID"
fi

CKPT_PATH="${OUTPUT_DOWNLOADED_DIR}/${SELECTED_MODEL_ID}/model.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ ERROR: Checkpoint not found at ${CKPT_PATH}"
    exit 1
fi
echo ""

# ─────────────────────────────────────────────────────
# 5. Fixed parameters
# ─────────────────────────────────────────────────────
NUM_TASKS=1
NUM_SEEDS=1
NUM_REPEATS=1

echo "Configuration summary:"
echo "  Dataset    : $SELECTED_DATASET (obs_dim=${STATE_DIM})"
echo "  Model      : $SELECTED_MODEL_ID"
echo "  Tasks      : $NUM_TASKS  Seeds: $NUM_SEEDS  Repeats: $NUM_REPEATS"
echo ""

# ─────────────────────────────────────────────────────
# 6. Execute Job Generator
# ─────────────────────────────────────────────────────
echo "Running: python3 generate_jobs_generalized.py --dataset $SELECTED_DATASET --model_id $SELECTED_MODEL_ID ..."
python3 generate_jobs_generalized.py \
    --dataset "$SELECTED_DATASET" \
    --model_id "$SELECTED_MODEL_ID" \
    --num_tasks "$NUM_TASKS" \
    --num_seeds "$NUM_SEEDS" \
    --num_repeats "$NUM_REPEATS"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Job generation failed!"
    exit 1
fi

echo ""
echo "Generation Complete."
echo ""
echo "====================================================="
echo "  Starting Job Execution via run_jobs.py"
echo "====================================================="
echo ""

python3 run_jobs.py | tee /tmp/mctd_run_jobs.log

JOB_EXIT_CODE=$?

echo ""
if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "====================================================="
    echo "  ✅ All Jobs Complete Successfully!"
    echo "====================================================="
else
    echo "====================================================="
    echo "  ❌ Jobs Failed with exit code: $JOB_EXIT_CODE"
    echo "====================================================="
    exit $JOB_EXIT_CODE
fi
