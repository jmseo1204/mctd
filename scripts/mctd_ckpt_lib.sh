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
MCTD_DOCKER_IMAGE="${MCTD_DOCKER_IMAGE:-$DOCKER_IMAGE}"
MCTD_DOCKER_USER="${MCTD_DOCKER_USER:-$DOCKER_USER}"
MCTD_DOCKER_OUTPUTS="/home/$MCTD_DOCKER_USER/mctd/outputs"
MCTD_OUTPUT_MOUNT_DIR="${MCTD_OUTPUT_MOUNT_DIR:-/home/$DOCKER_USER/mctd/outputs}"
MCTD_EVAL_BASE="$MCTD_OUTPUT_MOUNT_DIR"
MCTD_DOWNLOADED_DIR="$MCTD_OUTPUT_MOUNT_DIR/downloaded/$WANDB_ENTITY/$WANDB_PROJECT"

# ── mctd_select_gpus ──────────────────────────────────────────────────────────
# Display a table of all detected GPUs (status, VRAM, running processes with
# user / PID / uptime), prompt the user to pick GPU index(es), then override
# the AVAILABLE_GPUS variable (and export it so child processes inherit it).
#
# Usage: mctd_select_gpus
# Side-effect: sets + exports AVAILABLE_GPUS="localhost:N[,localhost:M,...]"
mctd_select_gpus() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[gpu-select] nvidia-smi not found — keeping AVAILABLE_GPUS=${AVAILABLE_GPUS:-localhost:0}"
        return 0
    fi

    local _n_gpus
    _n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if ! [[ "$_n_gpus" =~ ^[0-9]+$ ]] || [ "$_n_gpus" -eq 0 ]; then
        echo "[gpu-select] No GPUs detected — keeping AVAILABLE_GPUS=${AVAILABLE_GPUS:-localhost:0}"
        return 0
    fi

    echo ""
    echo "====================================================="
    echo "  GPU Status"
    echo "====================================================="
    printf "  %-4s  %-10s  %-28s  %s\n" "IDX" "STATUS" "NAME" "VRAM (Used / Total)"
    local _bar; _bar=$(printf '─%.0s' {1..74})
    printf "  %s\n" "$_bar"

    local _idx _info _name _mem_used _mem_total _procs _proc_line
    local _pid _pname _pmem _psinfo _user _etime
    for _idx in $(seq 0 $((_n_gpus - 1))); do
        _info=$(nvidia-smi -i "$_idx" \
            --query-gpu=name,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null) || continue

        _name=$(    echo "$_info" | cut -d',' -f1 | sed 's/^ *//')
        _mem_used=$(echo "$_info" | cut -d',' -f2 | sed 's/^ *//;s/ MiB//')
        _mem_total=$(echo "$_info"| cut -d',' -f3 | sed 's/^ *//;s/ MiB//')

        _procs=$(nvidia-smi -i "$_idx" \
            --query-compute-apps=pid,process_name,used_gpu_memory \
            --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') || true

        local _status
        [ -n "$_procs" ] && _status="OCCUPIED" || _status="FREE"
        [ "${#_name}" -gt 27 ] && _name="${_name:0:24}..."

        printf "  %-4s  %-10s  %-28s  %s / %s MiB\n" \
            "$_idx" "$_status" "$_name" "$_mem_used" "$_mem_total"

        if [ -n "$_procs" ]; then
            while IFS= read -r _proc_line; do
                [ -z "$_proc_line" ] && continue
                _pid=$(  echo "$_proc_line" | cut -d',' -f1 | tr -d ' ')
                _pname=$(echo "$_proc_line" | cut -d',' -f2 | sed 's/^ *//')
                _pmem=$( echo "$_proc_line" | cut -d',' -f3 | sed 's/^ *//')

                _psinfo=$(ps -p "$_pid" -o user=,etime= 2>/dev/null | head -1) || true
                _user=$( echo "$_psinfo" | awk '{print $1}')
                _etime=$(echo "$_psinfo" | awk '{print $2}')
                [ -z "$_user"  ] && _user="?"
                [ -z "$_etime" ] && _etime="?"

                printf "       ↳ PID %-7s  %-18s  user: %-12s  uptime: %-10s  %s\n" \
                    "$_pid" "$_pname" "$_user" "$_etime" "$_pmem"
            done <<< "$_procs"
        fi
    done

    printf "  %s\n" "$_bar"
    echo ""
    echo "  Current: AVAILABLE_GPUS=${AVAILABLE_GPUS:-localhost:0}"
    echo "  (Press Enter to keep current, Ctrl-C to abort)"
    echo ""
    read -rp "  Select GPU index(es) [space- or comma-separated, e.g. '0' or '0 1']: " _input

    local _indices
    _indices=$(echo "$_input" | tr ',' ' ' | tr -s ' ' | sed 's/^ //;s/ $//')
    if [ -z "$_indices" ]; then
        echo "  → Keeping AVAILABLE_GPUS=${AVAILABLE_GPUS:-localhost:0}"
        echo ""
        export AVAILABLE_GPUS="${AVAILABLE_GPUS:-localhost:0}"
        return 0
    fi

    # Validate indices and build new value
    local _new_gpus="" _idx_in
    for _idx_in in $_indices; do
        if ! [[ "$_idx_in" =~ ^[0-9]+$ ]]; then
            echo "  ❌ Invalid index '$_idx_in' — keeping current selection"
            export AVAILABLE_GPUS="${AVAILABLE_GPUS:-localhost:0}"
            return 0
        fi
        if [ "$_idx_in" -ge "$_n_gpus" ]; then
            echo "  ❌ Index $_idx_in out of range (0–$((_n_gpus - 1))) — keeping current selection"
            export AVAILABLE_GPUS="${AVAILABLE_GPUS:-localhost:0}"
            return 0
        fi
        _new_gpus="${_new_gpus:+$_new_gpus,}localhost:${_idx_in}"
    done

    AVAILABLE_GPUS="$_new_gpus"
    export AVAILABLE_GPUS
    echo "  ✓ AVAILABLE_GPUS set to: $AVAILABLE_GPUS"
    echo ""
}

# ── mctd_check_gpu_availability ───────────────────────────────────────────────
# Check that every localhost GPU listed in AVAILABLE_GPUS is free.
# Call AFTER killing any previous MCTD containers so they don't false-positive.
# Exits the calling script with status 1 if any GPU is occupied.
#
# Usage: mctd_check_gpu_availability
mctd_check_gpu_availability() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[gpu-check] nvidia-smi not found — skipping GPU availability check"
        return 0
    fi

    echo "Checking GPU availability (AVAILABLE_GPUS=${AVAILABLE_GPUS:-localhost:0})..."

    local _old_ifs="$IFS"
    local _entry _host _gpu _procs
    local _any_occupied=0

    IFS=","
    for _entry in ${AVAILABLE_GPUS:-localhost:0}; do
        IFS="$_old_ifs"
        _host="${_entry%%:*}"
        _gpu="${_entry##*:}"

        # Only check GPUs on the local machine
        if [ "$_host" != "localhost" ]; then
            IFS=","
            continue
        fi

        _procs="$(nvidia-smi -i "$_gpu" \
            --query-compute-apps=pid,process_name,used_gpu_memory \
            --format=csv,noheader 2>/dev/null)" || true

        # Only flag as occupied if total process VRAM exceeds threshold (MiB).
        # Background/system processes (e.g. desktop compositor, WSL2 services)
        # typically use < 500 MiB and don't actually block GPU compute.
        local _GPU_BUSY_THRESHOLD_MIB="${GPU_BUSY_THRESHOLD_MIB:-1000}"
        local _total_proc_mem=0 _heavy_procs=""
        if [ -n "$_procs" ]; then
            while IFS= read -r _line; do
                [ -z "$_line" ] && continue
                local _pmib
                _pmib=$(echo "$_line" | cut -d',' -f3 | sed 's/[^0-9]//g')
                _pmib="${_pmib:-0}"
                _total_proc_mem=$(( _total_proc_mem + _pmib ))
            done <<< "$_procs"
            if [ "$_total_proc_mem" -ge "$_GPU_BUSY_THRESHOLD_MIB" ]; then
                _heavy_procs="$_procs"
            fi
        fi

        if [ -n "$_heavy_procs" ]; then
            _any_occupied=1
            echo "  ✗ GPU $_gpu is occupied (${_total_proc_mem} MiB ≥ threshold ${_GPU_BUSY_THRESHOLD_MIB} MiB):"
            echo "$_heavy_procs" | while IFS= read -r _line; do
                echo "      $_line"
            done
        elif [ -n "$_procs" ]; then
            echo "  ✓ GPU $_gpu is free (${_total_proc_mem} MiB background processes ignored, threshold ${_GPU_BUSY_THRESHOLD_MIB} MiB)"
        else
            echo "  ✓ GPU $_gpu is free"
        fi

        IFS=","
    done
    IFS="$_old_ifs"

    if [ "$_any_occupied" -ne 0 ]; then
        echo ""
        echo "❌ ERROR: One or more GPUs are occupied — cannot start."
        echo "   Kill the processes above, or update AVAILABLE_GPUS in:"
        echo "   $MCTD_PROJECT_DIR/scripts/project_config.sh"
        exit 1
    fi

    echo ""
}

# ── mctd_dim_menu ─────────────────────────────────────────────────────────────
# [LEGACY] Prompts user to pick obs dim.  Kept for backward compatibility with
# old scripts; new train.sh reads these values from train_df_planning.yaml.
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
    echo "Obs dim: $MCTD_TARGET_OBS_DIM"
    echo "Dataset: $MCTD_DATASET_CONFIG"
}

# ── mctd_scan_ckpts [target_obs_dim] ─────────────────────────────────────────
# Scans MCTD_OUTPUT_MOUNT_DIR via Docker for model.ckpt files.
#
# Optional target_obs_dim (>0): filter by obs_dim. 0 or omitted: show all.
#
# Uses scripts/ckpt_scanner.py which provides:
#   - mtime-based metadata cache (skips torch.load on unchanged files)
#   - realpath deduplication (symlinks and originals show as one entry)
#
# Fills global MCTD_CKPT_DIRS with 12-field pipe-delimited entries:
#   name|model_id|host_path|mtime|epoch|obs_dim_indices|jump|dataset|frame_stack|network_size|num_layers|attn_heads
#   obs_dim_indices: compact JSON list e.g. [0,1] — never a bare count
mctd_scan_ckpts() {
    local target_dim="${1:-0}"
    MCTD_CKPT_DIRS=()

    if [ ! -d "$MCTD_OUTPUT_MOUNT_DIR" ] || [ -z "$(ls -A "$MCTD_OUTPUT_MOUNT_DIR" 2>/dev/null)" ]; then
        echo "❌ ERROR: Outputs directory not found or empty: $MCTD_OUTPUT_MOUNT_DIR" >&2
        echo "   Expected: ~/mctd/outputs" >&2
        return 1
    fi

    local scanner_host="$MCTD_PROJECT_DIR/scripts/ckpt_scanner.py"
    local scanner_docker="/tmp/ckpt_scanner.py"
    local scanner_project_docker="/tmp/mctd_project"

    local -a _raw
    mapfile -t _raw < <(
        docker run --rm \
            --entrypoint python3 \
            -v "$MCTD_OUTPUT_MOUNT_DIR":"$MCTD_DOCKER_OUTPUTS" \
            -v "$scanner_host":"$scanner_docker":ro \
            -v "$MCTD_PROJECT_DIR":"$scanner_project_docker":ro \
            "$MCTD_DOCKER_IMAGE" \
            "$scanner_docker" "$MCTD_DOCKER_OUTPUTS" "$MCTD_OUTPUT_MOUNT_DIR" "$target_dim" "$scanner_project_docker" \
            2>/dev/null | grep -E '^[^|]+(\|[^|]*){4,}'
    )

    for _entry in "${_raw[@]:-}"; do
        [ -n "$_entry" ] && MCTD_CKPT_DIRS+=("$_entry") || true
    done
}

# ── mctd_ckpt_menu [--no-fresh] ──────────────────────────────────────────────
# Displays checkpoint list (sorted by recency) with extended arch metadata,
# then prompts the user to select one.
#
# Reads global MCTD_CKPT_DIRS (populated by mctd_scan_ckpts).
# Each entry has 12 fields:
#   name|model_id|host_path|mtime|epoch|obs_dim_indices|jump|dataset|frame_stack|network_size|num_layers|attn_heads
#   obs_dim_indices: compact JSON list e.g. [0,1] or [0,1,...,28]  (never a bare count)
#
# Sets globals:
#   MCTD_SELECTED_CKPT             host path of selected model.ckpt (empty = fresh)
#   MCTD_SELECTED_EPOCH            epoch number of selected ckpt
#   MCTD_SELECTED_MODEL_ID         model_id extracted directly from scanner output
#   MCTD_SELECTED_DATASET          dataset config of selected ckpt (may be "unknown")
#   MCTD_SELECTED_OBS_DIM_INDICES  obs_dim_indices JSON of selected ckpt (may be "unknown")
#
# --no-fresh: hides "[0] Start from scratch" (use for eval scripts)
#             returns 1 if user enters invalid index
mctd_ckpt_menu() {
    local no_fresh="${1:-}"

    MCTD_SELECTED_CKPT=""
    MCTD_SELECTED_EPOCH=0
    MCTD_SELECTED_MODEL_ID=""
    MCTD_SELECTED_DATASET=""
    MCTD_SELECTED_OBS_DIM_INDICES=""

    if [ ${#MCTD_CKPT_DIRS[@]} -eq 0 ]; then
        echo "No checkpoints found."
        return 0
    fi

    # Sort by mtime (field 4) descending
    local -a _sorted
    mapfile -t _sorted < <(printf '%s\n' "${MCTD_CKPT_DIRS[@]}" | sort -t'|' -k4 -rn)

    echo ""
    echo "========================================"
    echo "Found ${#_sorted[@]} checkpoint(s):"
    echo "========================================"
    [ "$no_fresh" != "--no-fresh" ] && echo "  [0] Start from scratch (fresh training)"

    local i
    for i in "${!_sorted[@]}"; do
        local _e _model_id _epoch _obs_dim_indices _obs_dim_count _jump _dataset _fs _net _layers _heads
        _e="${_sorted[$i]}"
        _model_id=$(        echo "$_e" | cut -d'|' -f2)
        _epoch=$(           echo "$_e" | cut -d'|' -f5)
        _obs_dim_indices=$( echo "$_e" | cut -d'|' -f6)
        _jump=$(            echo "$_e" | cut -d'|' -f7)
        _dataset=$(         echo "$_e" | cut -d'|' -f8)
        _fs=$(              echo "$_e" | cut -d'|' -f9)
        _net=$(             echo "$_e" | cut -d'|' -f10)
        _layers=$(          echo "$_e" | cut -d'|' -f11)
        _heads=$(           echo "$_e" | cut -d'|' -f12)
        _epoch="${_epoch:-0}"
        # Derive display count from JSON indices list (e.g. [0,1] → 2)
        _obs_dim_count=$(python3 -c "import sys,json; v='$_obs_dim_indices'; print(len(json.loads(v)) if v!='unknown' else '?')" 2>/dev/null || echo "?")
        printf "  [%d] %-20s  dataset=%-38s obs=%s  jump=%s  fs=%s  net=%sL%sh%s  (epoch %s)\n" \
            "$((i+1))" "$_model_id" "$_dataset" "$_obs_dim_count" "$_jump" "$_fs" \
            "$_net" "$_layers" "$_heads" "$_epoch"
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
    MCTD_SELECTED_MODEL_ID=$(         echo "$_e" | cut -d'|' -f2)
    MCTD_SELECTED_CKPT=$(             echo "$_e" | cut -d'|' -f3)
    MCTD_SELECTED_EPOCH=$(            echo "$_e" | cut -d'|' -f5)
    MCTD_SELECTED_OBS_DIM_INDICES=$(  echo "$_e" | cut -d'|' -f6)
    MCTD_SELECTED_DATASET=$(          echo "$_e" | cut -d'|' -f8)
    MCTD_SELECTED_EPOCH="${MCTD_SELECTED_EPOCH:-0}"
    echo "Selected: $MCTD_SELECTED_CKPT  (epoch $MCTD_SELECTED_EPOCH)"
    echo "Model ID: $MCTD_SELECTED_MODEL_ID"
    return 0
}

# ── mctd_ensure_eval_symlink <host_ckpt_path> <model_id> ─────────────────────
# Ensures MCTD_DOWNLOADED_DIR/<model_id>/model.ckpt exists.
# If the checkpoint is already under MCTD_DOWNLOADED_DIR/<model_id>/, nothing
# is done (downloaded ckpts are already in the right place).
# For local (non-downloaded) checkpoints, creates a RELATIVE symlink so the
# path resolves correctly both on the host and inside Docker.
# Safe to call multiple times (ln -sf is idempotent).
mctd_ensure_eval_symlink() {
    local host_ckpt="$1"
    local model_id="$2"
    local real_ckpt target_dir
    real_ckpt=$(realpath "$host_ckpt" 2>/dev/null || echo "$host_ckpt")
    target_dir="$MCTD_DOWNLOADED_DIR/$model_id"

    # Already in the canonical downloaded location — nothing to do
    if [ "$(realpath "$target_dir/model.ckpt" 2>/dev/null)" = "$real_ckpt" ]; then
        return 0
    fi

    # Local checkpoint: create symlink inside downloaded dir
    mkdir -p "$target_dir"
    local rel_path
    rel_path=$(realpath --relative-to="$target_dir" "$real_ckpt")
    ln -sf "$rel_path" "$target_dir/model.ckpt"
}
