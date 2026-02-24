#!/bin/bash
# Wrapper script to run jobs with comprehensive monitoring

set -euo pipefail

PROJECT_DIR=$(pwd)
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Initialize log files
CHECKPOINT_LOG="$LOGS_DIR/checkpoints.txt"
STDERR_LOG="$LOGS_DIR/stderr.log"
MEMORY_LOG="$LOGS_DIR/memory_monitor.log"

echo "[WRAPPER] Clearing old logs..."
> "$CHECKPOINT_LOG"
> "$STDERR_LOG"
> "$MEMORY_LOG"

echo "[WRAPPER] ============================================================" | tee -a "$CHECKPOINT_LOG"
echo "[WRAPPER] Starting experiment with comprehensive monitoring" | tee -a "$CHECKPOINT_LOG"
echo "[WRAPPER] Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHECKPOINT_LOG"
echo "[WRAPPER] ============================================================" | tee -a "$CHECKPOINT_LOG"

echo "[WRAPPER] Logging locations:"
echo "[WRAPPER]   - Main logs: $LOGS_DIR/validation_run.jsonl"
echo "[WRAPPER]   - Checkpoints: $CHECKPOINT_LOG"
echo "[WRAPPER]   - Stderr: $STDERR_LOG"
echo "[WRAPPER]   - Memory: $MEMORY_LOG"

# Start memory monitor in background
echo "[WRAPPER] Starting memory monitor background process..."
python3 scripts/memory_monitor.py > /dev/null 2>&1 &
MONITOR_PID=$!
echo "[WRAPPER] Memory monitor PID: $MONITOR_PID" | tee -a "$CHECKPOINT_LOG"

# Cleanup function
cleanup() {
    echo "[WRAPPER] Caught signal at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHECKPOINT_LOG"
    echo "[WRAPPER] Killing memory monitor (PID: $MONITOR_PID)..." | tee -a "$CHECKPOINT_LOG"
    kill $MONITOR_PID 2>/dev/null || true
    echo "[WRAPPER] Cleanup complete" | tee -a "$CHECKPOINT_LOG"
}

trap cleanup EXIT

# Run main job
echo "[WRAPPER] Executing: python3 ./run_jobs.py" | tee -a "$CHECKPOINT_LOG"
echo "[WRAPPER] Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHECKPOINT_LOG"

python3 ./run_jobs.py 2>"$STDERR_LOG" || {
    EXIT_CODE=$?
    echo "[WRAPPER] Job failed with exit code: $EXIT_CODE" | tee -a "$CHECKPOINT_LOG"
    echo "[WRAPPER] End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHECKPOINT_LOG"
    exit $EXIT_CODE
}

echo "[WRAPPER] Job completed successfully" | tee -a "$CHECKPOINT_LOG"
echo "[WRAPPER] End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHECKPOINT_LOG"
