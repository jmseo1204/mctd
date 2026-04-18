#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 8 ]; then
    echo "Usage: $0 <results_dir> <num_repeats> <num_tasks> <waypoint_top_n> <summary_output> [dataset_name] [run_timestamp] [task_override_snapshot]" >&2
    exit 1
fi

RESULTS_DIR="$1"
NUM_REPEATS="$2"
NUM_TASKS="$3"
WAYPOINT_TOP_N="$4"
SUMMARY_OUTPUT="$5"
DATASET_NAME="${6:-}"
RUN_TIMESTAMP="${7:-}"
TASK_OVERRIDE_SNAPSHOT="${8:-}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

export AVAILABLE_GPUS="${AVAILABLE_GPUS:-}"
export MCTD_RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

PYTHONUNBUFFERED=1 python3 scripts/run_jobs.py

COLLECT_ARGS=(
    --results_dir "$RESULTS_DIR"
    --expected_repeats "$NUM_REPEATS"
    --expected_tasks "$NUM_TASKS"
    --expected_waypoint_top_n "$WAYPOINT_TOP_N"
    --summary_output "$SUMMARY_OUTPUT"
)

if [ -n "$DATASET_NAME" ]; then
    COLLECT_ARGS+=(--dataset_name "$DATASET_NAME")
fi
if [ -n "$RUN_TIMESTAMP" ]; then
    COLLECT_ARGS+=(--run_timestamp "$RUN_TIMESTAMP")
fi
if [ -n "$TASK_OVERRIDE_SNAPSHOT" ]; then
    COLLECT_ARGS+=(--task_override_snapshot "$TASK_OVERRIDE_SNAPSHOT")
fi

PYTHONUNBUFFERED=1 python3 scripts/collect_hamiltonian_benchmark_results.py \
    "${COLLECT_ARGS[@]}"

echo ""
echo "Hamiltonian benchmark results stored under: $PROJECT_DIR/$RESULTS_DIR"
echo "Hamiltonian benchmark summary JSON: $PROJECT_DIR/$SUMMARY_OUTPUT"
