#!/usr/bin/env bash
# scripts/analyze_logs.sh
# Usage: bash scripts/analyze_logs.sh <path/to/logfile.jsonl>
#
# Analyzes a jsonl log file and generates an HTML report.
# Output: reports/<logfile_stem>_analysis.html

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYZE_PY="$SCRIPT_DIR/analyze_logs.py"

# ── 인자 검증 ─────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/analyze_logs.sh <path/to/logfile.jsonl>"
    echo ""
    echo "Example:"
    echo "  bash scripts/analyze_logs.sh logs/run_20250221_143022.jsonl"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    # Try relative to project root
    if [ -f "$PROJECT_ROOT/$LOG_FILE" ]; then
        LOG_FILE="$PROJECT_ROOT/$LOG_FILE"
    else
        echo "Error: log file not found: $LOG_FILE"
        exit 1
    fi
fi

# ── Python 환경 확인 ──────────────────────────────────────────────────────────
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: python or python3 not found in PATH"
    exit 1
fi

PYTHON=$(command -v python3 2>/dev/null || command -v python)

# ── 의존성 확인 및 설치 ───────────────────────────────────────────────────────
echo "[analyze_logs] Checking dependencies..."
$PYTHON -c "import plotly, pandas" 2>/dev/null || {
    echo "[analyze_logs] Installing required packages: plotly pandas..."
    $PYTHON -m pip install plotly pandas --quiet
}

# ── 분석 실행 ─────────────────────────────────────────────────────────────────
echo "[analyze_logs] Analyzing: $LOG_FILE"

REPORT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORT_DIR"

$PYTHON "$ANALYZE_PY" \
    --log-file "$LOG_FILE" \
    --output-dir "$REPORT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    STEM=$(basename "$LOG_FILE" .jsonl)
    REPORT_PATH="$REPORT_DIR/${STEM}_analysis.html"
    echo ""
    echo "=========================================="
    echo " Report generated:"
    echo "   $REPORT_PATH"
    echo "=========================================="
    echo ""
    echo "Open the report in your browser, then ask the agent for suggestions."
else
    echo "Error: analysis failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
