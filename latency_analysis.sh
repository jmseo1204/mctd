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
    # Collect up to 5 most recent files that have timing data.
    mapfile -t CANDIDATES < <(
        find "$LOG_DIR" -maxdepth 1 -name "timestamp_anal_*.jsonl" -print 2>/dev/null \
            | sort -r \
            | while IFS= read -r f; do
                grep -q '"timing\.' "$f" 2>/dev/null && echo "$f"
            done \
            | head -5
    )

    if [ ${#CANDIDATES[@]} -eq 0 ]; then
        echo "Error: no timestamp_anal_*.jsonl files with timing data found in $LOG_DIR/"
        echo "  (run a job first, or pass a log path explicitly)"
        exit 1
    fi

    echo ""
    echo "Recent log files with timing data:"
    for i in "${!CANDIDATES[@]}"; do
        f="${CANDIDATES[$i]}"
        n_records=$(wc -l < "$f" 2>/dev/null || echo "?")
        printf "  [%d] %s  (%s lines)\n" "$((i+1))" "$(basename "$f")" "$n_records"
    done
    echo ""

    if [ ${#CANDIDATES[@]} -eq 1 ]; then
        LOG_FILE="${CANDIDATES[0]}"
        echo "Only one file found — using: $(basename "$LOG_FILE")" >&2
    else
        printf "Select log [1-%d] (default 1, most recent): " "${#CANDIDATES[@]}"
        read -r _choice </dev/tty
        _choice="${_choice:-1}"
        if ! [[ "$_choice" =~ ^[0-9]+$ ]] || [ "$_choice" -lt 1 ] || [ "$_choice" -gt "${#CANDIDATES[@]}" ]; then
            echo "Invalid selection '$_choice', defaulting to 1"
            _choice=1
        fi
        LOG_FILE="${CANDIDATES[$((_choice-1))]}"
        echo "[latency_analysis] Selected: $(basename "$LOG_FILE")" >&2
    fi
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
