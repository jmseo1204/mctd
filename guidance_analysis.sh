#!/usr/bin/env bash
# guidance_analysis.sh
# Analyzes validation_anal_*.jsonl logs (guidance quality, MCTS debug data).
#
# Usage:
#   bash guidance_analysis.sh                           # auto-selects most recent validation_anal_*.jsonl
#   bash guidance_analysis.sh <path/to/logfile.jsonl>   # uses specified file

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_PY="$PROJECT_ROOT/scripts/guidance_analysis.py"
LOG_DIR="$PROJECT_ROOT/logs"

if [ $# -lt 1 ]; then
    LATEST=$(find "$LOG_DIR" -maxdepth 1 -name "validation_anal_*.jsonl" -print 2>/dev/null \
        | sort -r | { head -1; cat > /dev/null; })
    if [ -z "$LATEST" ]; then
        echo "Error: no validation_anal_*.jsonl files found in $LOG_DIR/"
        echo "  (run a job first, or pass a log path explicitly)"
        exit 1
    fi
    LOG_FILE="$LATEST"
    echo "[guidance_analysis] No file specified, using most recent: $(basename "$LOG_FILE")" >&2
else
    LOG_FILE="$1"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: log file not found: $LOG_FILE"
    exit 1
fi

PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [ -z "$PYTHON" ]; then
    echo "Error: python or python3 not found in PATH"
    exit 1
fi

echo "[guidance_analysis] Analyzing: $(basename "$LOG_FILE")"
echo "=========================================="
"$PYTHON" "$ANALYZE_PY" "$LOG_FILE"
echo "=========================================="
echo "  Analysis Complete"
echo "=========================================="
