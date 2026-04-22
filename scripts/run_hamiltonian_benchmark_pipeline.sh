#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
if [ "$MODE" = "single" ]; then
    if [ "$#" -lt 6 ] || [ "$#" -gt 10 ]; then
        echo "Usage: $0 single <results_dir> <num_repeats> <num_tasks> <waypoint_top_n> <summary_output> [dataset_name] [run_timestamp] [task_override_snapshot] [n_per_bin]" >&2
        exit 1
    fi
    RESULTS_DIR="$2"
    NUM_REPEATS="$3"
    NUM_TASKS="$4"
    WAYPOINT_TOP_N="$5"
    SUMMARY_OUTPUT="$6"
    DATASET_NAME="${7:-}"
    RUN_TIMESTAMP="${8:-}"
    TASK_OVERRIDE_SNAPSHOT="${9:-}"
    N_PER_BIN="${10:-}"
elif [ "$MODE" = "dual" ]; then
    if [ "$#" -lt 9 ] || [ "$#" -gt 13 ]; then
        echo "Usage: $0 dual <online_results_dir> <baseline_results_dir> <num_repeats> <num_tasks> <waypoint_top_n> <online_summary_output> <baseline_summary_output> <comparison_output> [dataset_name] [run_timestamp] [task_override_snapshot] [n_per_bin]" >&2
        exit 1
    fi
    ONLINE_RESULTS_DIR="$2"
    BASELINE_RESULTS_DIR="$3"
    NUM_REPEATS="$4"
    NUM_TASKS="$5"
    WAYPOINT_TOP_N="$6"
    ONLINE_SUMMARY_OUTPUT="$7"
    BASELINE_SUMMARY_OUTPUT="$8"
    COMPARISON_OUTPUT="$9"
    DATASET_NAME="${10:-}"
    RUN_TIMESTAMP="${11:-}"
    TASK_OVERRIDE_SNAPSHOT="${12:-}"
    N_PER_BIN="${13:-}"
else
    if [ "$#" -lt 5 ] || [ "$#" -gt 9 ]; then
        echo "Usage: $0 <results_dir> <num_repeats> <num_tasks> <waypoint_top_n> <summary_output> [dataset_name] [run_timestamp] [task_override_snapshot] [n_per_bin]" >&2
        exit 1
    fi
    MODE="single"
    RESULTS_DIR="$1"
    NUM_REPEATS="$2"
    NUM_TASKS="$3"
    WAYPOINT_TOP_N="$4"
    SUMMARY_OUTPUT="$5"
    DATASET_NAME="${6:-}"
    RUN_TIMESTAMP="${7:-}"
    TASK_OVERRIDE_SNAPSHOT="${8:-}"
    N_PER_BIN="${9:-}"
fi

# Derive N_PER_BIN from WAYPOINT_TOP_N if not provided (assumes 3 bins).
if [ -z "$N_PER_BIN" ] || [ "$N_PER_BIN" = "0" ]; then
    N_PER_BIN=$(( WAYPOINT_TOP_N / 3 ))
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

export AVAILABLE_GPUS="${AVAILABLE_GPUS:-}"
export MCTD_RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

PYTHONUNBUFFERED=1 python3 scripts/run_jobs.py

build_collect_args() {
    local results_dir="$1"
    local summary_output="$2"
    local -a collect_args=(
        --results_dir "$results_dir"
        --expected_repeats "$NUM_REPEATS"
        --expected_tasks "$NUM_TASKS"
        --expected_waypoint_top_n "$WAYPOINT_TOP_N"
        --n_per_bin "$N_PER_BIN"
        --summary_output "$summary_output"
    )

    if [ -n "$DATASET_NAME" ]; then
        collect_args+=(--dataset_name "$DATASET_NAME")
    fi
    if [ -n "$RUN_TIMESTAMP" ]; then
        collect_args+=(--run_timestamp "$RUN_TIMESTAMP")
    fi
    if [ -n "$TASK_OVERRIDE_SNAPSHOT" ]; then
        collect_args+=(--task_override_snapshot "$TASK_OVERRIDE_SNAPSHOT")
    fi

    printf '%s\n' "${collect_args[@]}"
}

if [ "$MODE" = "single" ]; then
    mapfile -t COLLECT_ARGS < <(build_collect_args "$RESULTS_DIR" "$SUMMARY_OUTPUT")
    PYTHONUNBUFFERED=1 python3 scripts/collect_hamiltonian_benchmark_results.py \
        "${COLLECT_ARGS[@]}"

    echo ""
    echo "Hamiltonian benchmark results stored under: $PROJECT_DIR/$RESULTS_DIR"
    echo "Hamiltonian benchmark summary JSON: $PROJECT_DIR/$SUMMARY_OUTPUT"
    exit 0
fi

mapfile -t ONLINE_COLLECT_ARGS < <(build_collect_args "$ONLINE_RESULTS_DIR" "$ONLINE_SUMMARY_OUTPUT")
PYTHONUNBUFFERED=1 python3 scripts/collect_hamiltonian_benchmark_results.py \
    "${ONLINE_COLLECT_ARGS[@]}"

mapfile -t BASELINE_COLLECT_ARGS < <(build_collect_args "$BASELINE_RESULTS_DIR" "$BASELINE_SUMMARY_OUTPUT")
PYTHONUNBUFFERED=1 python3 scripts/collect_hamiltonian_benchmark_results.py \
    "${BASELINE_COLLECT_ARGS[@]}"

PYTHONUNBUFFERED=1 python3 scripts/collect_hamiltonian_benchmark_comparison.py \
    --online_summary "$ONLINE_SUMMARY_OUTPUT" \
    --baseline_summary "$BASELINE_SUMMARY_OUTPUT" \
    --output "$COMPARISON_OUTPUT"

COMPARISON_MD_OUTPUT="${COMPARISON_OUTPUT%.json}.md"
PYTHONUNBUFFERED=1 python3 scripts/render_hamiltonian_benchmark_comparison_md.py \
    --comparison_summary "$COMPARISON_OUTPUT" \
    --output_md "$COMPARISON_MD_OUTPUT"

echo ""
echo "Hamiltonian online results stored under: $PROJECT_DIR/$ONLINE_RESULTS_DIR"
echo "Hamiltonian baseline results stored under: $PROJECT_DIR/$BASELINE_RESULTS_DIR"
echo "Hamiltonian online summary JSON: $PROJECT_DIR/$ONLINE_SUMMARY_OUTPUT"
echo "Hamiltonian baseline summary JSON: $PROJECT_DIR/$BASELINE_SUMMARY_OUTPUT"
echo "Hamiltonian comparison summary JSON: $PROJECT_DIR/$COMPARISON_OUTPUT"
echo "Hamiltonian comparison markdown: $PROJECT_DIR/$COMPARISON_MD_OUTPUT"
