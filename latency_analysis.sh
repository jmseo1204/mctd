#!/usr/bin/env bash
# latency_analysis.sh
# Usage:
#   bash latency_analysis.sh                        # auto-selects most recently modified jsonl
#   bash latency_analysis.sh <path/to/logfile.jsonl> # uses specified file

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_PY="$PROJECT_ROOT/scripts/latency_analysis.py"
LOG_DIR="$PROJECT_ROOT/logs"

if [ $# -lt 1 ]; then
    # Pick the most recent (by filename timestamp) file that has timing data.
    # Interrupted jobs produce tiny files with no timing.* records — skip them.
    LATEST=$(find "$LOG_DIR" -maxdepth 1 -name "timestamp_anal_*.jsonl" -print 2>/dev/null \
        | sort -r \
        | while IFS= read -r f; do
            grep -q '"timing\.' "$f" 2>/dev/null && echo "$f" && break
        done)
    if [ -z "$LATEST" ]; then
        echo "Error: no timestamp_anal_*.jsonl files with timing data found in $LOG_DIR/"
        echo "  (run a job first, or pass a log path explicitly)"
        exit 1
    fi
    LOG_FILE="$LATEST"
    echo "[latency_analysis] No file specified, using most recent with timing data: $(basename "$LOG_FILE")" >&2
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

"$PYTHON" "$ANALYZE_PY" --log-file "$LOG_FILE"
