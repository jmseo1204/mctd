#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 8 ]; then
    echo "Usage: $0 <results_dir> <num_repeats> <num_tasks> <rollouts_per_task> [summary_output] [dataset_name] [run_timestamp] [jobs_dir]" >&2
    exit 1
fi

RESULTS_DIR="$1"
NUM_REPEATS="$2"
NUM_TASKS="$3"
ROLLOUTS_PER_TASK="$4"
SUMMARY_OUTPUT="${5:-}"
DATASET_NAME="${6:-}"
RUN_TIMESTAMP="${7:-}"
JOBS_DIR="${8:-jobs}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

export AVAILABLE_GPUS="${AVAILABLE_GPUS:-}"
RUN_LOG_TAG="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
if [ -n "$DATASET_NAME" ]; then
    RUN_LOG_TAG="${DATASET_NAME}_${RUN_LOG_TAG}"
fi
export MCTD_RUN_TIMESTAMP="$RUN_LOG_TAG"
export MCTD_JOBS_DIR="$JOBS_DIR"

PYTHONUNBUFFERED=1 python3 scripts/run_jobs.py

COLLECT_ARGS=(
    --results_dir "$RESULTS_DIR"
    --expected_repeats "$NUM_REPEATS"
    --expected_tasks "$NUM_TASKS"
    --expected_rollouts "$ROLLOUTS_PER_TASK"
)

if [ -n "$SUMMARY_OUTPUT" ]; then
    COLLECT_ARGS+=(--summary_output "$SUMMARY_OUTPUT")
fi
if [ -n "$DATASET_NAME" ]; then
    COLLECT_ARGS+=(--dataset_name "$DATASET_NAME")
fi
if [ -n "$RUN_TIMESTAMP" ]; then
    COLLECT_ARGS+=(--run_timestamp "$RUN_TIMESTAMP")
fi

PYTHONUNBUFFERED=1 python3 scripts/collect_benchmark_results.py \
    "${COLLECT_ARGS[@]}"

echo ""
echo "Benchmark results stored under: $PROJECT_DIR/$RESULTS_DIR"
if [ -n "$SUMMARY_OUTPUT" ]; then
    echo "Benchmark summary JSON: $PROJECT_DIR/$SUMMARY_OUTPUT"
fi
