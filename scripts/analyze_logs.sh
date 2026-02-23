#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYZE_PY="$SCRIPT_DIR/analyze_logs.py"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/analyze_logs.sh <path/to/logfile.jsonl>"
    echo "Example: bash scripts/analyze_logs.sh logs_memory_debug/interact_1771868054.jsonl"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: log file not found: $LOG_FILE"
    exit 1
fi

if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: python or python3 not found"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

echo "[analyze_logs] Analyzing: $LOG_FILE"

REPORT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORT_DIR"

$PYTHON "$ANALYZE_PY" --log-file "$LOG_FILE" --output-dir "$REPORT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    STEM=$(basename "$LOG_FILE" .jsonl)
    REPORT_PATH="$REPORT_DIR/${STEM}_analysis.html"
    echo ""
    echo "=========================================="
    echo " Report generated:"
    echo "   $REPORT_PATH"
    echo "=========================================="
else
    echo "Error: analysis failed"
    exit $EXIT_CODE
fi
