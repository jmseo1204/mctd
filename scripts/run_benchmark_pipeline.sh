#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <results_dir> <num_repeats> <num_tasks> <rollouts_per_task>" >&2
    exit 1
fi

RESULTS_DIR="$1"
NUM_REPEATS="$2"
NUM_TASKS="$3"
ROLLOUTS_PER_TASK="$4"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

export AVAILABLE_GPUS="${AVAILABLE_GPUS:-}"

python3 scripts/run_jobs.py

python3 scripts/collect_benchmark_results.py \
    --results_dir "$RESULTS_DIR" \
    --expected_repeats "$NUM_REPEATS" \
    --expected_tasks "$NUM_TASKS" \
    --expected_rollouts "$ROLLOUTS_PER_TASK"

echo ""
echo "Benchmark results stored under: $PROJECT_DIR/$RESULTS_DIR"
