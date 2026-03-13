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

# ── Shared constants (callers may override before sourcing) ──────────────────
MCTD_DOCKER_IMAGE="${MCTD_DOCKER_IMAGE:-mctd:0.1}"
MCTD_DOCKER_USER="${MCTD_DOCKER_USER:-jmseo1204}"
MCTD_DOCKER_OUTPUTS="/home/$MCTD_DOCKER_USER/mctd/outputs"
MCTD_OUTPUT_MOUNT_DIR="${MCTD_OUTPUT_MOUNT_DIR:-/home/jmseo1204/mctd_outputs}"
MCTD_EVAL_BASE="$MCTD_OUTPUT_MOUNT_DIR/downloaded/jmseo1204-seoul-national-university/mctd_eval"

# ── mctd_ckpt_map_file <dim> ─────────────────────────────────────────────────
# Returns path to the checkpoint→model_id map file for the given obs dim.
mctd_ckpt_map_file() {
    echo "$MCTD_PROJECT_DIR/logs/${1}d_ckpt_model_map.txt"
}

# ── mctd_dim_menu ─────────────────────────────────────────────────────────────
# Prompts user to pick obs dim.
# Sets globals: MCTD_TARGET_OBS_DIM  MCTD_DATASET_CONFIG
mctd_dim_menu() {
    echo "Select observation state dimension:"
    echo "  1) 2D  (x,y position only)   [og_antmaze_giant_navigate]"
    echo "  2) 15D (qpos only)            [og_antmaze_giant_navigate_15d]"
    echo "  3) 29D (full qpos+qvel)       [og_antmaze_giant_navigate_fullstate]"
    echo ""
    read -rp "Enter [1-3]: " _mctd_dim_sel
    case "$_mctd_dim_sel" in
        1) MCTD_TARGET_OBS_DIM=2;  MCTD_DATASET_CONFIG="og_antmaze_giant_navigate"            ;;
        2) MCTD_TARGET_OBS_DIM=15; MCTD_DATASET_CONFIG="og_antmaze_giant_navigate_15d"       ;;
        3) MCTD_TARGET_OBS_DIM=29; MCTD_DATASET_CONFIG="og_antmaze_giant_navigate_fullstate" ;;
        *) echo "Invalid selection '$_mctd_dim_sel'. Exiting." >&2; return 1 ;;
    esac
    echo "✓ Selected state dim: $MCTD_TARGET_OBS_DIM"
    echo "✓ Selected dataset:   $MCTD_DATASET_CONFIG"
}

# ── mctd_scan_ckpts <target_obs_dim> ─────────────────────────────────────────
# Scans MCTD_OUTPUT_MOUNT_DIR via Docker for model.ckpt files whose
# data_mean.shape[0] matches target_obs_dim.
# Covers both training outputs and mctd_eval downloaded checkpoints.
# Fills global MCTD_CKPT_DIRS with entries: "name|host_path|mtime|epoch"
mctd_scan_ckpts() {
    local target_dim="$1"
    MCTD_CKPT_DIRS=()

    local scanner
    scanner=$(mktemp /tmp/mctd_scan_XXXXXX.py)
    chmod 644 "$scanner"
    cat > "$scanner" <<'PYEOF'
import sys, torch
from pathlib import Path

docker_outputs   = sys.argv[1]   # Docker-internal path to outputs/ (= MCTD_OUTPUT_MOUNT_DIR)
output_mount_dir = sys.argv[2]   # Corresponding host path
target_dim       = int(sys.argv[3])

base = Path(docker_outputs)
if not base.exists():
    print(f"ERROR: {docker_outputs} not found inside container", file=sys.stderr)
    sys.exit(0)

for ckpt_path in sorted(base.rglob("model.ckpt")):
    is_mctd_eval = "mctd_eval" in ckpt_path.parts
    is_downloaded = "downloaded" in ckpt_path.parts

    # Skip raw downloaded dirs that are not organized under mctd_eval
    if is_downloaded and not is_mctd_eval:
        continue

    try:
        # Resolve symlink so torch.load reads the real file
        real_path = ckpt_path.resolve()
        ckpt  = torch.load(str(real_path), map_location="cpu", weights_only=False)
        sd    = ckpt.get("state_dict", {})
        dm    = sd.get("data_mean")
        if dm is None or int(dm.shape[0]) != target_dim:
            continue
        epoch = ckpt.get("epoch", 0)
        mtime = int(real_path.stat().st_mtime)
        # host_path points to the symlink (or real file) within OUTPUT_MOUNT_DIR
        host_path = output_mount_dir + str(ckpt_path)[len(docker_outputs):]

        if is_mctd_eval:
            model_id = ckpt_path.parent.name
            name = f"[eval] {model_id}"
        else:
            parts = ckpt_path.parts
            try:
                idx  = parts.index("outputs") + 1
                name = "/".join(parts[idx:idx+2])
            except (ValueError, IndexError):
                name = str(ckpt_path.parent)

        print(f"{name}|{host_path}|{mtime}|{epoch}")
    except Exception as e:
        print(f"SCAN_ERR {ckpt_path}: {e}", file=sys.stderr)
PYEOF

    local -a _raw
    mapfile -t _raw < <(
        docker run --rm \
            --entrypoint python3 \
            -v "$MCTD_OUTPUT_MOUNT_DIR":"$MCTD_DOCKER_OUTPUTS" \
            -v "$scanner":"$scanner":ro \
            "$MCTD_DOCKER_IMAGE" \
            "$scanner" "$MCTD_DOCKER_OUTPUTS" "$MCTD_OUTPUT_MOUNT_DIR" "$target_dim" \
            2>/dev/null | grep -E '^[^|]+\|[^|]+\|[0-9]+\|[0-9]+$'
    )
    rm -f "$scanner"

    for _entry in "${_raw[@]:-}"; do
        [ -n "$_entry" ] && MCTD_CKPT_DIRS+=("$_entry")
    done
}

# ── mctd_ckpt_menu <dim> [--no-fresh] ────────────────────────────────────────
# Displays checkpoint list sorted by recency and prompts for selection.
# Reads global MCTD_CKPT_DIRS (populated by mctd_scan_ckpts).
# Sets globals:
#   MCTD_SELECTED_CKPT        host path of selected model.ckpt (empty = fresh)
#   MCTD_SELECTED_EPOCH       epoch number of selected ckpt
#   MCTD_SELECTED_MODEL_ID    model_id if known, empty otherwise
#
# --no-fresh: hides "[0] Start from scratch" (use for eval scripts)
#             returns 1 if user enters invalid index
mctd_ckpt_menu() {
    local dim="$1"
    local no_fresh="${2:-}"
    local ckpt_map_file
    ckpt_map_file=$(mctd_ckpt_map_file "$dim")

    MCTD_SELECTED_CKPT=""
    MCTD_SELECTED_EPOCH=0
    MCTD_SELECTED_MODEL_ID=""

    if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
        echo "No existing ${dim}D checkpoints found."
        return 0
    fi

    local -a _sorted
    mapfile -t _sorted < <(printf '%s\n' "${MCTD_CKPT_DIRS[@]}" | sort -t'|' -k3 -rn)

    echo ""
    echo "========================================"
    echo "Found ${#_sorted[@]} checkpoint(s) with obs_dim=${dim}:"
    echo "========================================"
    [ "$no_fresh" != "--no-fresh" ] && echo "  [0] Start from scratch (fresh training)"

    local i
    for i in "${!_sorted[@]}"; do
        local _e _ckpt_name _ckpt_path _epoch_num _ckpt_dir _mapped _disp _hint
        _e="${_sorted[$i]}"
        _ckpt_name=$(echo "$_e" | cut -d'|' -f1)
        _ckpt_path=$(echo "$_e" | cut -d'|' -f2)
        _epoch_num=$(echo "$_e" | cut -d'|' -f4)
        _epoch_num="${_epoch_num:-0}"
        [ -z "$_ckpt_path" ] && continue
        _ckpt_dir=$(dirname "$(realpath "$_ckpt_path" 2>/dev/null || echo "$_ckpt_path")")
        _mapped=""
        if [ -f "$ckpt_map_file" ]; then
            _mapped=$(grep "^$_ckpt_dir|" "$ckpt_map_file" | cut -d'|' -f2 | tail -1 || true)
        fi
        if [ -n "$_mapped" ]; then
            _disp="$_mapped"
            _hint="  [$_ckpt_name]"
        else
            _disp="$_ckpt_name"
            _hint=""
        fi
        printf "  [%d] %s  (epoch %s)%s\n" "$((i+1))" "$_disp" "$_epoch_num" "$_hint"
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
    local _ckpt_name _ckpt_dir _mapped
    MCTD_SELECTED_CKPT=$(echo "$_e"  | cut -d'|' -f2)
    MCTD_SELECTED_EPOCH=$(echo "$_e" | cut -d'|' -f4)
    MCTD_SELECTED_EPOCH="${MCTD_SELECTED_EPOCH:-0}"
    _ckpt_name=$(echo "$_e" | cut -d'|' -f1)
    _ckpt_dir=$(dirname "$(realpath "$MCTD_SELECTED_CKPT" 2>/dev/null || echo "$MCTD_SELECTED_CKPT")")
    _mapped=""
    if [ -f "$ckpt_map_file" ]; then
        _mapped=$(grep "^$_ckpt_dir|" "$ckpt_map_file" | cut -d'|' -f2 | tail -1 || true)
    fi
    if [ -n "$_mapped" ]; then
        MCTD_SELECTED_MODEL_ID="$_mapped"
    elif [[ "$_ckpt_name" == train_* ]]; then
        MCTD_SELECTED_MODEL_ID="$_ckpt_name"
    fi
    echo "Selected: $MCTD_SELECTED_CKPT  (epoch $MCTD_SELECTED_EPOCH)"
    [ -n "$MCTD_SELECTED_MODEL_ID" ] && echo "Model ID: $MCTD_SELECTED_MODEL_ID"
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
