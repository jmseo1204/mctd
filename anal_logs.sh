#!/bin/bash

# Integrated Analysis Script
# This script allows interactive selection of recent logs for analysis

LOG_DIR="logs"

echo "=========================================="
echo "    Experiment Log Analysis Tool"
echo "=========================================="
echo ""

if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Directory '$LOG_DIR' not found."
    exit 1
fi

# Get 5 most recent log files sorted by modification time
mapfile -t RECENT_LOGS < <(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -n 5)

if [ ${#RECENT_LOGS[@]} -eq 0 ]; then
    echo "No .log files found in '$LOG_DIR'."
    exit 1
fi

echo "Recent logs in $LOG_DIR/:"
for i in "${!RECENT_LOGS[@]}"; do
    filename=$(basename "${RECENT_LOGS[$i]}")
    # Mark the default (most recent)
    if [ $i -eq 0 ]; then
        echo "[$i] $filename (default)"
    else
        echo "[$i] $filename"
    fi
done

echo ""
read -p "Select log index to analyze (0-$((${#RECENT_LOGS[@]} - 1)), default: 0): " idx

# Default to 0 if input is empty
idx=${idx:-0}

# Validate input
if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 0 ] && [ "$idx" -lt "${#RECENT_LOGS[@]}" ]; then
    SELECTED_LOG="${RECENT_LOGS[$idx]}"
    echo ""
    echo "Analyzing: $(basename "$SELECTED_LOG")"
    echo "=========================================="
    python3 analyze_guidance_results.py "$SELECTED_LOG"
else
    echo "Invalid selection."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Analysis Complete"
echo "=========================================="
