#!/usr/bin/env bash
# train_dql.sh
# Interactive Docker launcher for dql/main_Antmaze.py

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/project_config.sh
source "$PROJECT_DIR/scripts/project_config.sh"

MCTD_PROJECT_DIR="$PROJECT_DIR"
# shellcheck source=scripts/mctd_ckpt_lib.sh
source "$PROJECT_DIR/scripts/mctd_ckpt_lib.sh"

DOCKER_PROJECT="/home/$DOCKER_USER/mctd"
RESULTS_DIR="$PROJECT_DIR/dql/results"
LOGS_DIR="$PROJECT_DIR/logs"
HOME_DIR="$HOME"
OGBENCH_DATA_DIR="$(cd "$PROJECT_DIR/../ogbench_data" 2>/dev/null && pwd || true)"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
mkdir -p "$HOME_DIR/.d4rl" "$HOME_DIR/.d4rl/datasets"

ENV_NAME=""
RESUME_DIR=""
RESUME_EPOCH=""
EXP_TAG=""
KILL_EXISTING=0
EXTRA_ARGS=()
DQL_AVAILABLE_ENVS=()

usage() {
    cat <<'EOF'
Usage: ./train_dql.sh [options] [extra main_Antmaze.py args]

Options:
  --env_name NAME         DQL dataset env name (discovered from ../ogbench_data/*.npz)
  --resume-dir PATH       Resume from an existing dql/results/<run_dir>
  --resume-epoch N        Resume from a specific saved epoch in that run dir
  --exp TAG               Experiment tag for fresh runs
  --kill-existing         Stop running mctd_dql_training_* containers first
  -h, --help              Show this help

Unrecognized options are passed through to dql/main_Antmaze.py.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        --resume-dir|--resume_dir)
            RESUME_DIR="$2"
            shift 2
            ;;
        --resume-epoch|--resume_epoch)
            RESUME_EPOCH="$2"
            shift 2
            ;;
        --exp)
            EXP_TAG="$2"
            shift 2
            ;;
        --kill-existing)
            KILL_EXISTING=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "$RESUME_DIR" ]; then
    if [[ "$RESUME_DIR" != /* ]]; then
        RESUME_DIR="$PROJECT_DIR/$RESUME_DIR"
    fi
    RESUME_DIR="$(realpath -m "$RESUME_DIR")"
fi

dql_discover_envs() {
    mapfile -t DQL_AVAILABLE_ENVS < <(
        find "$OGBENCH_DATA_DIR" -maxdepth 1 -type f -name 'antmaze-*-v0.npz' \
            | sed 's#^.*/##' \
            | sed 's/\.npz$//' \
            | LC_ALL=C sort -u
    )

    if [ ${#DQL_AVAILABLE_ENVS[@]} -eq 0 ]; then
        echo "ERROR: no antmaze dataset files found in $OGBENCH_DATA_DIR" >&2
        echo "Expected files like antmaze-giant-stitch-v0.npz" >&2
        exit 1
    fi
}

dql_env_supported() {
    local candidate="$1"
    local _env
    for _env in "${DQL_AVAILABLE_ENVS[@]}"; do
        if [ "$_env" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

dql_print_available_envs() {
    local _env
    for _env in "${DQL_AVAILABLE_ENVS[@]}"; do
        echo "  - $_env" >&2
    done
}

dql_dataset_menu() {
    local count="${#DQL_AVAILABLE_ENVS[@]}"
    local i

    if [ "$count" -eq 1 ]; then
        ENV_NAME="${DQL_AVAILABLE_ENVS[0]}"
        echo "[dataset] Only one antmaze dataset found in $OGBENCH_DATA_DIR: $ENV_NAME"
        return 0
    fi

    echo "Select DQL dataset from $OGBENCH_DATA_DIR:"
    for i in "${!DQL_AVAILABLE_ENVS[@]}"; do
        printf "  %d) %s\n" "$((i + 1))" "${DQL_AVAILABLE_ENVS[$i]}"
    done
    echo ""
    read -rp "Enter [1-$count]: " _sel
    if ! [[ "$_sel" =~ ^[0-9]+$ ]] || [ "$_sel" -lt 1 ] || [ "$_sel" -gt "$count" ]; then
        echo "Invalid selection '$_sel'." >&2
        exit 1
    fi
    ENV_NAME="${DQL_AVAILABLE_ENVS[$((_sel - 1))]}"
}

dql_detect_env_from_resume_dir() {
    python3 - "$1" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
variant_path = os.path.join(run_dir, "variant.json")
env_name = ""
if os.path.exists(variant_path):
    try:
        with open(variant_path) as f:
            env_name = json.load(f).get("env_name", "")
    except Exception:
        env_name = ""
if not env_name:
    base = os.path.basename(os.path.normpath(run_dir))
    if "|" in base:
        env_name = base.split("|", 1)[0]
print(env_name)
PY
}

dql_latest_epoch() {
    python3 - "$1" <<'PY'
import os, re, sys
run_dir = sys.argv[1]
pattern = re.compile(r"actor_(\d+)\.pth$")
epochs = []
if os.path.isdir(run_dir):
    for entry in os.listdir(run_dir):
        match = pattern.fullmatch(entry)
        if match:
            epochs.append(int(match.group(1)))
print(max(epochs) if epochs else "")
PY
}

dql_scan_resume_runs() {
    python3 - "$RESULTS_DIR" "$1" <<'PY'
import datetime as dt
import json
import os
import re
import sys

root, env_name = sys.argv[1], sys.argv[2]
pattern = re.compile(r"actor_(\d+)\.pth$")
runs = []

if os.path.isdir(root):
    for name in os.listdir(root):
        if env_name and not name.startswith(env_name + "|"):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue

        latest_epoch = None
        for entry in os.listdir(path):
            match = pattern.fullmatch(entry)
            if match is None:
                continue
            epoch = int(match.group(1))
            latest_epoch = epoch if latest_epoch is None else max(latest_epoch, epoch)
        if latest_epoch is None:
            continue

        variant = {}
        variant_path = os.path.join(path, "variant.json")
        if os.path.exists(variant_path):
            try:
                with open(variant_path) as f:
                    variant = json.load(f)
            except Exception:
                variant = {}

        updated_at = dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        fields = [
            name,
            path,
            str(latest_epoch),
            updated_at,
            str(variant.get("seed", "?")),
            str(variant.get("exp", "?")),
        ]
        print("\t".join(fields))
PY
}

dql_resume_menu() {
    mapfile -t DQL_RUNS < <(dql_scan_resume_runs "$ENV_NAME")
    if [ ${#DQL_RUNS[@]} -eq 0 ]; then
        echo "No existing DQL runs found for $ENV_NAME. Starting fresh."
        return 0
    fi

    echo ""
    echo "========================================"
    echo "Existing DQL runs for $ENV_NAME"
    echo "========================================"
    echo "  [0] Start from scratch"

    local i entry name path latest updated seed exp
    for i in "${!DQL_RUNS[@]}"; do
        entry="${DQL_RUNS[$i]}"
        IFS=$'\t' read -r name path latest updated seed exp <<< "$entry"
        printf "  [%d] exp=%-18s latest=%-4s seed=%-3s updated=%s\n" \
            "$((i + 1))" "$exp" "$latest" "$seed" "$updated"
        printf "      %s\n" "$name"
    done
    echo ""

    read -rp "Select run to resume [0-${#DQL_RUNS[@]}]: " _sel
    if [ -z "$_sel" ] || [ "$_sel" = "0" ]; then
        return 0
    fi
    if ! [[ "$_sel" =~ ^[0-9]+$ ]] || [ "$_sel" -lt 1 ] || [ "$_sel" -gt ${#DQL_RUNS[@]} ]; then
        echo "Invalid selection '$_sel'." >&2
        exit 1
    fi

    entry="${DQL_RUNS[$((_sel - 1))]}"
    IFS=$'\t' read -r _name RESUME_DIR RESUME_EPOCH _updated _seed _exp <<< "$entry"
}

host_to_docker_path() {
    local p="$1"
    if [[ "$p" != /* ]]; then
        p="$PROJECT_DIR/$p"
    fi
    p="$(realpath -m "$p")"
    if [[ "$p" == "$PROJECT_DIR"* ]]; then
        echo "${DOCKER_PROJECT}${p#$PROJECT_DIR}"
    else
        echo "$p"
    fi
}

pick_single_gpu() {
    local first_local=""
    IFS=',' read -r -a _gpu_entries <<< "${AVAILABLE_GPUS:-localhost:0}"
    for _entry in "${_gpu_entries[@]}"; do
        local host="${_entry%%:*}"
        local gpu="${_entry##*:}"
        if [ "$host" = "localhost" ]; then
            first_local="$gpu"
            break
        fi
    done

    if [ -z "$first_local" ]; then
        echo "ERROR: DQL launcher requires a localhost GPU selection." >&2
        exit 1
    fi

    if [ ${#_gpu_entries[@]} -gt 1 ]; then
        echo "[gpu] DQL training is single-GPU; using localhost:${first_local}."
    fi

    AVAILABLE_GPUS="localhost:${first_local}"
    export AVAILABLE_GPUS
    DQL_GPU_ID="$first_local"
}

if [ ! -d "$OGBENCH_DATA_DIR" ]; then
    echo "ERROR: OGBench data directory not found: $PROJECT_DIR/../ogbench_data" >&2
    exit 1
fi

dql_discover_envs

echo "===================================================="
echo "  DQL Training Launcher"
echo "===================================================="

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed or not in PATH" >&2
    exit 1
fi
if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running" >&2
    exit 1
fi

# Keep repo files writable after Docker writes as root.
docker run --rm --user root -v "$PROJECT_DIR":"$DOCKER_PROJECT" "$DOCKER_IMAGE" \
    chmod -R a+rwX "$DOCKER_PROJECT/dql/results" "$DOCKER_PROJECT/logs" >/dev/null 2>&1 || true

mctd_select_gpus
pick_single_gpu
mctd_check_gpu_availability

if [ "$KILL_EXISTING" -eq 1 ]; then
    echo "========================================"
    echo "[$(date)] --kill-existing: stopping all running mctd_dql_training_* containers..."
    docker ps --filter "name=mctd_dql_training_" --format "{{.Names}}" | xargs -r docker rm -f >/dev/null 2>&1 || true
    sleep 1
    echo "Done."
fi

if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo "ERROR: resume directory not found: $RESUME_DIR" >&2
        exit 1
    fi
    if [ -z "$ENV_NAME" ]; then
        ENV_NAME="$(dql_detect_env_from_resume_dir "$RESUME_DIR")"
    fi
fi

if [ -z "$ENV_NAME" ]; then
    dql_dataset_menu
fi

if ! dql_env_supported "$ENV_NAME"; then
    echo "ERROR: unsupported env_name '$ENV_NAME'" >&2
    echo "Available antmaze datasets in $OGBENCH_DATA_DIR:" >&2
    dql_print_available_envs
    exit 1
fi

if [ -z "$RESUME_DIR" ]; then
    dql_resume_menu
fi

if [ -n "$RESUME_DIR" ]; then
    if [ -z "$RESUME_EPOCH" ]; then
        RESUME_EPOCH="$(dql_latest_epoch "$RESUME_DIR")"
    fi
    if [ -z "$RESUME_EPOCH" ]; then
        echo "ERROR: no actor_{epoch}.pth files found in $RESUME_DIR" >&2
        exit 1
    fi
    if [ ! -f "$RESUME_DIR/actor_${RESUME_EPOCH}.pth" ]; then
        echo "ERROR: missing actor_${RESUME_EPOCH}.pth in $RESUME_DIR" >&2
        exit 1
    fi
else
    DEFAULT_EXP="exp_$(date +%Y%m%d_%H%M%S)"
    if [ -z "$EXP_TAG" ]; then
        read -rp "Experiment tag [$DEFAULT_EXP]: " EXP_TAG
    fi
    EXP_TAG="${EXP_TAG// /_}"
    [ -z "$EXP_TAG" ] && EXP_TAG="$DEFAULT_EXP"
fi

RUN_LABEL="$ENV_NAME"
if [ -n "$RESUME_DIR" ]; then
    RUN_LABEL="$(basename "$RESUME_DIR")"
fi
SAFE_LABEL="$(echo "$RUN_LABEL" | tr '/| ' '_' | tr -cd '[:alnum:]_.-')"
LOG_FILE="$LOGS_DIR/train_dql_${SAFE_LABEL}.log"
CONTAINER_NAME="mctd_dql_training_$$"

trap_handler() {
    local exit_code=$?
    local signal="${1:-EXIT}"
    if [ "$signal" = "EXIT" ] && [ "$exit_code" -eq 0 ]; then
        return
    fi
    echo "[$(date)] [FATAL] train_dql.sh terminated. signal=$signal exit_code=$exit_code line=$BASH_LINENO" | tee -a "$LOG_FILE"
}
trap 'trap_handler EXIT' EXIT
trap 'trap_handler INT; exit 130' INT
trap 'trap_handler TERM; exit 143' TERM
trap 'trap_handler ERR' ERR

PY_ARGS=(python3 dql/main_Antmaze.py --env_name "$ENV_NAME" --device 0)
if [ -n "$RESUME_DIR" ]; then
    PY_ARGS+=(--resume_dir "$(host_to_docker_path "$RESUME_DIR")" --resume_epoch "$RESUME_EPOCH")
else
    PY_ARGS+=(--exp "$EXP_TAG")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

printf -v PY_CMD '%q ' "${PY_ARGS[@]}"
INNER_SCRIPT="cd $DOCKER_PROJECT && git config --global --add safe.directory $DOCKER_PROJECT 2>/dev/null; mkdir -p $DOCKER_PROJECT/dql/results $DOCKER_PROJECT/logs; $PY_CMD"

NETRC_MOUNT=()
if [ -f "$HOME_DIR/.netrc" ]; then
    NETRC_MOUNT=(-v "$HOME_DIR/.netrc:/home/$DOCKER_USER/.netrc")
fi

DOCKER_CMD=(
    docker run -d --gpus all --user root
    --name "$CONTAINER_NAME"
    --shm-size=50g
    -e "CUDA_VISIBLE_DEVICES=$DQL_GPU_ID"
    -e "HOME=/home/$DOCKER_USER"
    -e "MUJOCO_GL=osmesa"
    -e "HYDRA_FULL_ERROR=1"
    -e "WANDB_ENTITY=$WANDB_ENTITY"
    -e "WANDB_PROJECT=$WANDB_PROJECT"
    -e "WANDB_MODE=$WANDB_MODE_VALUE"
    -e "D4RL_DATASET_DIR=/home/$DOCKER_USER/.d4rl/datasets"
    -e "LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/$DOCKER_USER/.mujoco/mujoco210/bin"
    -v "/usr/lib/wsl:/usr/lib/wsl"
    -v "$PROJECT_DIR:$DOCKER_PROJECT"
    -v "$OGBENCH_DATA_DIR:/home/$DOCKER_USER/.ogbench/data"
    -v "$HOME_DIR/.d4rl:/home/$DOCKER_USER/.d4rl"
    "${NETRC_MOUNT[@]}"
    "$DOCKER_IMAGE" /bin/bash -lc "$INNER_SCRIPT"
)

echo "========================================" | tee -a "$LOG_FILE"
echo "[$(date)] Starting DQL AntMaze training" | tee -a "$LOG_FILE"
echo "Project dir : $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Docker image: $DOCKER_IMAGE" | tee -a "$LOG_FILE"
echo "Dataset     : $ENV_NAME" | tee -a "$LOG_FILE"
echo "GPU         : localhost:$DQL_GPU_ID" | tee -a "$LOG_FILE"
echo "WandB mode  : $WANDB_MODE_VALUE" | tee -a "$LOG_FILE"
if [ -n "$RESUME_DIR" ]; then
    echo "Resume dir  : $RESUME_DIR" | tee -a "$LOG_FILE"
    echo "Resume epoch: $RESUME_EPOCH" | tee -a "$LOG_FILE"
else
    echo "Exp tag     : $EXP_TAG" | tee -a "$LOG_FILE"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra args  : ${EXTRA_ARGS[*]}" | tee -a "$LOG_FILE"
fi
printf -v DOCKER_CMD_PRETTY '%q ' "${DOCKER_CMD[@]}"
echo "[$(date)] Docker command: $DOCKER_CMD_PRETTY" | tee -a "$LOG_FILE"

set +e
CONTAINER_ID="$("${DOCKER_CMD[@]}" 2>&1)"
RUN_RC=$?
set -e

if [ $RUN_RC -ne 0 ]; then
    echo "[$(date)] Failed to start container: $CONTAINER_ID" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[$(date)] Container started: $CONTAINER_NAME ($CONTAINER_ID)" | tee -a "$LOG_FILE"
echo "[$(date)] Monitor logs : tail -f $LOG_FILE" | tee -a "$LOG_FILE"
echo "[$(date)] Live logs    : docker logs -f $CONTAINER_NAME" | tee -a "$LOG_FILE"

docker logs -f "$CONTAINER_NAME" >> "$LOG_FILE" 2>&1 &
STREAM_PID=$!

EXIT_CODE="$(docker wait "$CONTAINER_NAME" 2>/dev/null || echo 1)"

kill "$STREAM_PID" 2>/dev/null || true
wait "$STREAM_PID" 2>/dev/null || true

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --rm --user root -v "$PROJECT_DIR":"$DOCKER_PROJECT" "$DOCKER_IMAGE" \
    chmod -R a+rwX "$DOCKER_PROJECT/dql/results" "$DOCKER_PROJECT/logs" >/dev/null 2>&1 || true

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[$(date)] Training completed successfully." | tee -a "$LOG_FILE"
    exit 0
fi

echo "[$(date)] Training failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
exit 1
