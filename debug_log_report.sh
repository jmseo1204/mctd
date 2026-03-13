#!/usr/bin/env bash
# debug_log_report.sh
# Usage: bash debug_log_report.sh <path/to/logfile.jsonl>

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE_PY="$PROJECT_ROOT/scripts/debug_log_report.py"

if [ $# -lt 1 ]; then
    echo "Usage: bash debug_log_report.sh <path/to/logfile.jsonl>"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: log file not found: $LOG_FILE"
    exit 1
fi

if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: python or python3 not found in PATH"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

echo "[debug_log_report] Checking dependencies..."
$PYTHON -c "import plotly, pandas" 2>/dev/null || {
    echo "[debug_log_report] Installing required packages..."
    $PYTHON -m pip install plotly pandas --quiet
}

echo "[debug_log_report] Analyzing: $LOG_FILE"
REPORT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORT_DIR"

$PYTHON "$ANALYZE_PY" --log-file "$LOG_FILE" --output-dir "$REPORT_DIR"

if [ $? -eq 0 ]; then
    STEM=$(basename "$LOG_FILE" .jsonl)
    REPORT_PATH="$REPORT_DIR/${STEM}_analysis.html"
    echo ""
    echo "=========================================="
    echo " Report generated:"
    echo "   $REPORT_PATH"
    echo "=========================================="
fi
