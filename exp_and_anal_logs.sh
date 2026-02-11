#!/bin/bash

# Integrated Experiment and Analysis Script
# This script runs the experiment and then allows interactive log analysis

LOG_DIR="experiment_logs"

echo "=========================================="
echo "  Experiment & Analysis Workflow"
echo "=========================================="
echo ""

# Step 1: Kill any running docker containers and jobs
echo "[1/3] Cleaning up existing containers and jobs..."
docker ps -q | xargs -r docker kill 2>/dev/null
pkill -f run_jobs.py 2>/dev/null
sleep 2
echo "Cleanup complete."
echo ""

# Step 2: Run the experiment
echo "[2/3] Starting experiment..."
PREP_OUTPUT=$(python3 run_experiment.py)

# Extract log path from python output
LOG_PATH=$(echo "$PREP_OUTPUT" | grep "LOG_FILE_PATH:" | cut -d':' -f2-)

if [ -z "$LOG_PATH" ]; then
    echo "Error: Could not determine log file path from run_experiment.py"
    exit 1
fi

echo "Log path initialized: $LOG_PATH"
# Clear file if exists
> "$LOG_PATH"

# Background process to harvest Docker logs into the log file while experiment runs
(
    # Set of tracked container IDs to avoid duplicate log tails
    tracked_containers=""
    
    # Stay alive as long as run_jobs.py is active
    while pgrep -f run_jobs.py > /dev/null; do
        # Find running containers
        CONT_IDS=$(docker ps --format "{{.ID}}")
        
        for CID in $CONT_IDS; do
            # If we haven't tracked this container yet, start a background tail
            if [[ ! "$tracked_containers" =~ "$CID" ]]; then
                tracked_containers="$tracked_containers $CID"
                # Start individual tail for this PID/CID in background
                # Appending to the main log file
                (docker logs -f "$CID" >> "$LOG_PATH" 2>&1) &
            fi
        done
        sleep 2
    done
) &

echo ""
echo "=========================================="
echo "        EXPERIMENT PROGRESS (TQDM)"
echo "=========================================="
echo " Running jobs. Terminal will show progress bar."
echo " Experiment logs are being saved to $LOG_PATH in background."
echo "------------------------------------------"

# Run the job engine in the foreground to show TQDM bar
python3 run_jobs.py

echo ""
echo "Experiment execution finished."

echo ""
echo ""
echo "=========================================="
echo "[3/3] Analysis Phase"
echo "=========================================="
echo ""

# Step 3: Interactive log selection for analysis
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Directory '$LOG_DIR' not found."
    exit 1
fi

LOGS=($(ls "$LOG_DIR"/*.log 2>/dev/null))

if [ ${#LOGS[@]} -eq 0 ]; then
    echo "No .log files found in '$LOG_DIR'."
    exit 1
fi

echo "Available logs in $LOG_DIR/:"
for i in "${!LOGS[@]}"; do
    filename=$(basename "${LOGS[$i]}")
    echo "[$i] $filename"
done

echo ""
read -p "Select log index to analyze (0-$((${#LOGS[@]} - 1))): " idx

# Validate input
if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 0 ] && [ "$idx" -lt "${#LOGS[@]}" ]; then
    SELECTED_LOG="${LOGS[$idx]}"
    echo ""
    echo "Analyzing: $(basename $SELECTED_LOG)"
    echo "=========================================="
    python3 analyze_guidance_results.py "$SELECTED_LOG"
else
    echo "Invalid selection."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Workflow Complete"
echo "=========================================="
