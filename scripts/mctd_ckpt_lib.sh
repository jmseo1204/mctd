#!/usr/bin/env bash
# scripts/mctd_ckpt_lib.sh
# Shared MCTD checkpoint scanning and selection utilities.
#
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/scripts/mctd_ckpt_lib.sh"
#
# Globals set by callers before sourcing (or auto-detected):
#   MCTD_PROJECT_DIR  — repo root  (default: one level above this file)
#   MCTD_OUTPUT_MOUNT_DIR — host path where Docker writes outputs
#   MCTD_DOCKER_IMAGE / MCTD_DOCKER_USER — Docker image name / user
#
# Functions:
#   mctd_dim_menu            → MCTD_TARGET_OBS_DIM, MCTD_DATASET_CONFIG
#   mctd_scan_ckpts <dim>    → MCTD_CKPT_DIRS[]
#   mctd_ckpt_menu <dim> [--no-fresh]
#                            → MCTD_SELECTED_CKPT, MCTD_SELECTED_EPOCH,
#                               MCTD_SELECTED_MODEL_ID
#   mctd_ensure_eval_symlink <host_ckpt> <model_id>
#                            → creates MCTD_EVAL_BASE/<model_id>/model.ckpt

# ── Auto-detect project root from lib location ───────────────────────────────
_MCTD_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCTD_PROJECT_DIR="${MCTD_PROJECT_DIR:-$(cd "$_MCTD_LIB_DIR/.." && pwd)}"

# ── Load central user config (sets DOCKER_USER) ───────────────────────────────
source "$_MCTD_LIB_DIR/project_config.sh"

# ── Shared constants (callers may override before sourcing) ──────────────────
MCTD_DOCKER_IMAGE="${MCTD_DOCKER_IMAGE:-mctd:0.1}"
MCTD_DOCKER_USER="${MCTD_DOCKER_USER:-$DOCKER_USER}"
MCTD_DOCKER_OUTPUTS="/home/$MCTD_DOCKER_USER/mctd/outputs"
MCTD_OUTPUT_MOUNT_DIR="${MCTD_OUTPUT_MOUNT_DIR:-/home/$DOCKER_USER/mctd_outputs}"
MCTD_EVAL_BASE="$MCTD_OUTPUT_MOUNT_DIR"

# ── mctd_dim_menu ─────────────────────────────────────────────────────────────
# Prompts user to pick obs dim.
# Sets globals: MCTD_TARGET_OBS_DIM  MCTD_DATASET_CONFIG
mctd_dim_menu() {
    echo "Select dataset:"
    echo "  1) 2D  navigate (x,y only)      [og_antmaze_giant_navigate]"
    echo "  2) 15D navigate (qpos only)     [og_antmaze_giant_navigate_15d]"
    echo "  3) 29D navigate (qpos+qvel)     [og_antmaze_giant_navigate_fullstate]"
    echo "  4) 2D  stitch   (x,y only)      [og_antmaze_giant_stitch]"
    echo ""
    read -rp "Enter [1-4]: " _mctd_dim_sel
    case "$_mctd_dim_sel" in
        1) MCTD_TARGET_OBS_DIM=2;  MCTD_DATASET_CONFIG="og_antmaze_giant_navigate"            ;;
        2) MCTD_TARGET_OBS_DIM=15; MCTD_DATASET_CONFIG="og_antmaze_giant_navigate_15d"       ;;
        3) MCTD_TARGET_OBS_DIM=29; MCTD_DATASET_CONFIG="og_antmaze_giant_navigate_fullstate" ;;
        4) MCTD_TARGET_OBS_DIM=2;  MCTD_DATASET_CONFIG="og_antmaze_giant_stitch"             ;;
        *) echo "Invalid selection '$_mctd_dim_sel'. Exiting." >&2; return 1 ;;
    esac
    echo "✓ Selected state dim: $MCTD_TARGET_OBS_DIM"
    echo "✓ Selected dataset:   $MCTD_DATASET_CONFIG"
}

# ── mctd_scan_ckpts <target_obs_dim> ─────────────────────────────────────────
# Scans MCTD_OUTPUT_MOUNT_DIR via Docker for model.ckpt files whose
# data_mean.shape[0] matches target_obs_dim.
#
# Uses scripts/ckpt_scanner.py which provides:
#   - mtime-based metadata cache (skips torch.load on unchanged files)
#   - realpath deduplication (symlinks and originals show as one entry)
#
# Fills global MCTD_CKPT_DIRS with entries: "name|model_id|host_path|mtime|epoch"
mctd_scan_ckpts() {
    local target_dim="$1"
    MCTD_CKPT_DIRS=()

    local scanner_host="$MCTD_PROJECT_DIR/scripts/ckpt_scanner.py"
    local scanner_docker="/tmp/ckpt_scanner.py"

    local -a _raw
    mapfile -t _raw < <(
        docker run --rm \
            --entrypoint python3 \
            -v "$MCTD_OUTPUT_MOUNT_DIR":"$MCTD_DOCKER_OUTPUTS" \
            -v "$scanner_host":"$scanner_docker":ro \
            "$MCTD_DOCKER_IMAGE" \
            "$scanner_docker" "$MCTD_DOCKER_OUTPUTS" "$MCTD_OUTPUT_MOUNT_DIR" "$target_dim" \
            2>/dev/null | grep -E '^[^|]+\|[^|]+\|[^|]+\|[0-9]+\|[0-9]+$'
    )

    for _entry in "${_raw[@]:-}"; do
        [ -n "$_entry" ] && MCTD_CKPT_DIRS+=("$_entry") || true
    done
}

# ── mctd_ckpt_menu <dim> [--no-fresh] ────────────────────────────────────────
# Displays checkpoint list sorted by recency and prompts for selection.
# Reads global MCTD_CKPT_DIRS (populated by mctd_scan_ckpts).
# Each entry has 5 fields: name|model_id|host_path|mtime|epoch
# Sets globals:
#   MCTD_SELECTED_CKPT        host path of selected model.ckpt (empty = fresh)
#   MCTD_SELECTED_EPOCH       epoch number of selected ckpt
#   MCTD_SELECTED_MODEL_ID    model_id extracted directly from scanner output
#
# --no-fresh: hides "[0] Start from scratch" (use for eval scripts)
#             returns 1 if user enters invalid index
mctd_ckpt_menu() {
    local dim="$1"
    local no_fresh="${2:-}"

    MCTD_SELECTED_CKPT=""
    MCTD_SELECTED_EPOCH=0
    MCTD_SELECTED_MODEL_ID=""

    if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
        echo "No existing ${dim}D checkpoints found."
        return 0
    fi

    # Sort by mtime (field 4) descending
    local -a _sorted
    mapfile -t _sorted < <(printf '%s\n' "${MCTD_CKPT_DIRS[@]}" | sort -t'|' -k4 -rn)

    echo ""
    echo "========================================"
    echo "Found ${#_sorted[@]} checkpoint(s) with obs_dim=${dim}:"
    echo "========================================"
    [ "$no_fresh" != "--no-fresh" ] && echo "  [0] Start from scratch (fresh training)"

    local i
    for i in "${!_sorted[@]}"; do
        local _e _ckpt_model_id _epoch_num
        _e="${_sorted[$i]}"
        _ckpt_model_id=$(echo "$_e" | cut -d'|' -f2)
        _epoch_num=$(echo "$_e"     | cut -d'|' -f5)
        _epoch_num="${_epoch_num:-0}"
        printf "  [%d] %s  (epoch %s)\n" "$((i+1))" "$_ckpt_model_id" "$_epoch_num"
    done
    echo ""

    local max_idx=${#_sorted[@]}
    local _sel
    if [ "$no_fresh" = "--no-fresh" ]; then
        read -rp "Select checkpoint [1-${max_idx}]: " _sel
        if ! [[ "$_sel" =~ ^[0-9]+$ ]] || [ "$_sel" -lt 1 ] || [ "$_sel" -gt "$max_idx" ]; then
            echo "Invalid selection '$_sel'. Exiting." >&2
            return 1
        fi
    else
        read -rp "Select checkpoint to resume [0-${max_idx}]: " _sel
        if [ "$_sel" = "0" ] || [ -z "$_sel" ]; then
            echo "Starting fresh training."
            return 0
        fi
        if ! [[ "$_sel" =~ ^[0-9]+$ ]] || [ "$_sel" -lt 1 ] || [ "$_sel" -gt "$max_idx" ]; then
            echo "Invalid selection '$_sel'. Starting fresh training."
            return 0
        fi
    fi

    local _idx=$((_sel - 1))
    local _e="${_sorted[$_idx]}"
    MCTD_SELECTED_MODEL_ID=$(echo "$_e" | cut -d'|' -f2)
    MCTD_SELECTED_CKPT=$(echo "$_e"     | cut -d'|' -f3)
    MCTD_SELECTED_EPOCH=$(echo "$_e"    | cut -d'|' -f5)
    MCTD_SELECTED_EPOCH="${MCTD_SELECTED_EPOCH:-0}"
    echo "Selected: $MCTD_SELECTED_CKPT  (epoch $MCTD_SELECTED_EPOCH)"
    echo "Model ID: $MCTD_SELECTED_MODEL_ID"
    return 0
}

# ── mctd_ensure_eval_symlink <host_ckpt_path> <model_id> ─────────────────────
# Creates MCTD_EVAL_BASE/<model_id>/model.ckpt → <host_ckpt_path> symlink.
# Uses a RELATIVE symlink so it resolves correctly both on the host and inside
# Docker (where MCTD_OUTPUT_MOUNT_DIR is mounted at a different absolute path).
# Safe to call multiple times (ln -sf is idempotent).
mctd_ensure_eval_symlink() {
    local host_ckpt="$1"
    local model_id="$2"
    local real_ckpt eval_dir rel_path
    real_ckpt=$(realpath "$host_ckpt" 2>/dev/null || echo "$host_ckpt")
    eval_dir="$MCTD_EVAL_BASE/$model_id"
    mkdir -p "$eval_dir"
    # Compute relative path from eval_dir to the actual checkpoint file
    rel_path=$(realpath --relative-to="$eval_dir" "$real_ckpt")
    ln -sf "$rel_path" "$eval_dir/model.ckpt"
}
