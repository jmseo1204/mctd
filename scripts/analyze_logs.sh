#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYZE_PY="$SCRIPT_DIR/analyze_logs.py"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/analyze_logs.sh <path/to/logfile.jsonl>"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: log file not found: $LOG_FILE"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

echo "[analyze_logs] Checking dependencies..."
$PYTHON -c "import plotly, pandas" 2>/dev/null || {
    echo "[analyze_logs] Installing required packages..."
    $PYTHON -m pip install plotly pandas --quiet
}

echo "[analyze_logs] Analyzing: $LOG_FILE"

REPORT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORT_DIR"

$PYTHON "$ANALYZE_PY" --log-file "$LOG_FILE" --output-dir "$REPORT_DIR"

STEM=$(basename "$LOG_FILE" .jsonl)
REPORT_PATH="$REPORT_DIR/${STEM}_analysis.html"
echo ""
echo "=========================================="
echo " Report generated:"
echo "   $REPORT_PATH"
echo "=========================================="
