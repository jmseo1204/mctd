#!/bin/bash

# Guidance Analysis Script
# Reads per-job jsonl logs (e.g. logs/20260318_195017_uzrq13fa.jsonl)
# and reports per-tree (forward/backward) guidance stats.

LOG_DIR="logs"

echo "=========================================="
echo "    Guidance Log Analysis Tool"
echo "=========================================="
echo ""

if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Directory '$LOG_DIR' not found."
    exit 1
fi

# Get 5 most recent per-job jsonl files (YYYYMMDD_HHMMSS_*.jsonl pattern)
mapfile -t RECENT_LOGS < <(ls -t "$LOG_DIR"/[0-9]*_*.jsonl 2>/dev/null | head -n 5)

if [ ${#RECENT_LOGS[@]} -eq 0 ]; then
    echo "No per-job jsonl files found in '$LOG_DIR'."
    echo "(Expected pattern: logs/YYYYMMDD_HHMMSS_<model_id>.jsonl)"
    echo "Run eval.sh first to generate logs."
    exit 1
fi

echo "Recent job logs in $LOG_DIR/:"
for i in "${!RECENT_LOGS[@]}"; do
    filename=$(basename "${RECENT_LOGS[$i]}")
    size=$(wc -l < "${RECENT_LOGS[$i]}" 2>/dev/null || echo "?")
    if [ $i -eq 0 ]; then
        echo "[$i] $filename  ($size records) (default)"
    else
        echo "[$i] $filename  ($size records)"
    fi
done

echo ""
read -p "Select log index to analyze (0-$((${#RECENT_LOGS[@]} - 1)), default: 0): " idx
idx=${idx:-0}

if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 0 ] && [ "$idx" -lt "${#RECENT_LOGS[@]}" ]; then
    SELECTED_LOG="${RECENT_LOGS[$idx]}"
    echo ""
    echo "Analyzing: $(basename "$SELECTED_LOG")"
    echo "=========================================="
    python3 scripts/guidance_analysis.py "$SELECTED_LOG"
else
    echo "Invalid selection."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Analysis Complete"
echo "=========================================="
