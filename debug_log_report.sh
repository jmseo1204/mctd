#!/usr/bin/env bash
# debug_log_report.sh
# Usage:
#   bash debug_log_report.sh                        # auto-selects most recent per-job jsonl
#   bash debug_log_report.sh <path/to/logfile.jsonl> # uses specified file

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_PY="$PROJECT_ROOT/scripts/debug_log_report.py"
LOG_DIR="$PROJECT_ROOT/logs"

if [ $# -lt 1 ]; then
    # Auto-select the most recent per-job jsonl (YYYYMMDD_HHMMSS_*.jsonl)
    LATEST=$(ls -t "$LOG_DIR"/[0-9]*_*.jsonl 2>/dev/null | head -n 1)
    if [ -z "$LATEST" ]; then
        echo "Error: no per-job jsonl files found in $LOG_DIR/"
        echo "Hint: run eval.sh first to generate logs/YYYYMMDD_HHMMSS_<model_id>.jsonl"
        exit 1
    fi
    LOG_FILE="$LATEST"
    echo "[debug_log_report] No file specified, using most recent: $(basename "$LOG_FILE")"
else
    LOG_FILE="$1"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: log file not found: $LOG_FILE"
    exit 1
fi

if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: python or python3 not found in PATH"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

echo "[debug_log_report] Analyzing: $LOG_FILE"
REPORT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORT_DIR"

$PYTHON "$ANALYZE_PY" --log-file "$LOG_FILE" --output-dir "$REPORT_DIR"

if [ $? -eq 0 ]; then
    STEM=$(basename "$LOG_FILE" .jsonl)
    REPORT_PATH="$REPORT_DIR/${STEM}_analysis.md"
    echo ""
    echo "=========================================="
    echo " Report generated:"
    echo "   $REPORT_PATH"
    echo "=========================================="
fi
