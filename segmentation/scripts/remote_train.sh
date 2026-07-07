#!/usr/bin/env bash
#
# Sync this project to a GPU server, launch training, and retrieve results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/remote_train.sh <start|status|logs|fetch> --host USER@HOST [options]

Actions:
  start    Sync the project, prepare the remote environment and data, then
           launch training in the background.
  status   Report whether the remote training process is still running.
  logs     Print the last 100 lines of the remote training log.
  fetch    Download remote runs/segment results into runs/remote/<host>/.

Connection options:
  --host HOST              SSH destination or ~/.ssh/config alias.
                           Can also be set with REMOTE_HOST.
  --remote-dir PATH        Remote project directory (default: kap-forceps-segmentation).
  --remote-python PATH     Python used to create the venv (default: python3).

Training options used by "start":
  --config PATH            Dataset YAML (default: configs/forceps_seg.yaml).
  --preprocess-preset NAME Prepare and train on this preprocessing preset.
  --preprocess-config PATH Presets YAML (default: configs/preprocessing.yaml).
  --model MODEL            YOLO checkpoint or model YAML (default: yolo11n-seg.pt).
  --epochs N               Training epochs (default: 100).
  --imgsz N                Training image size (default: 1024).
  --batch N                Batch size (default: 8).
  --device DEVICE          CUDA device accepted by Ultralytics (default: 0).
  --patience N             Early-stopping patience (default: 20).
  --name NAME              Ultralytics run name (default: forceps_remote).
  --skip-sync              Reuse files already present on the server.

Fetch options:
  --local-results PATH     Download destination (default: runs/remote/<host>).

Examples:
  scripts/remote_train.sh start --host gpu-box --epochs 150 --batch 16
  scripts/remote_train.sh start --host gpu-box --preprocess-preset roi_clahe
  scripts/remote_train.sh status --host gpu-box
  scripts/remote_train.sh logs --host gpu-box
  scripts/remote_train.sh fetch --host gpu-box
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 2
fi

ACTION="$1"
shift
case "${ACTION}" in
    start|status|logs|fetch) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        usage >&2
        exit 2
        ;;
esac

HOST="${REMOTE_HOST:-}"
REMOTE_DIR="kap-forceps-segmentation"
REMOTE_PYTHON="python3"
DATASET_CONFIG="configs/forceps_seg.yaml"
PREPROCESS_PRESET=""
PREPROCESS_CONFIG="configs/preprocessing.yaml"
MODEL="yolo11n-seg.pt"
EPOCHS="100"
IMGSZ="1024"
BATCH="8"
DEVICE="0"
PATIENCE="20"
RUN_NAME="forceps_remote"
SKIP_SYNC="false"
LOCAL_RESULTS=""

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Option $1 requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            require_value "$@"
            HOST="$2"
            shift 2
            ;;
        --remote-dir)
            require_value "$@"
            REMOTE_DIR="$2"
            shift 2
            ;;
        --remote-python)
            require_value "$@"
            REMOTE_PYTHON="$2"
            shift 2
            ;;
        --config)
            require_value "$@"
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --preprocess-preset)
            require_value "$@"
            PREPROCESS_PRESET="$2"
            shift 2
            ;;
        --preprocess-config)
            require_value "$@"
            PREPROCESS_CONFIG="$2"
            shift 2
            ;;
        --model)
            require_value "$@"
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            require_value "$@"
            EPOCHS="$2"
            shift 2
            ;;
        --imgsz)
            require_value "$@"
            IMGSZ="$2"
            shift 2
            ;;
        --batch)
            require_value "$@"
            BATCH="$2"
            shift 2
            ;;
        --device)
            require_value "$@"
            DEVICE="$2"
            shift 2
            ;;
        --patience)
            require_value "$@"
            PATIENCE="$2"
            shift 2
            ;;
        --name)
            require_value "$@"
            RUN_NAME="$2"
            shift 2
            ;;
        --skip-sync)
            SKIP_SYNC="true"
            shift
            ;;
        --local-results)
            require_value "$@"
            LOCAL_RESULTS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${HOST}" ]]; then
    echo "--host or REMOTE_HOST is required." >&2
    exit 2
fi
if [[ "${REMOTE_DIR}" == "~/"* ]]; then
    REMOTE_DIR="${REMOTE_DIR#\~/}"
fi
if [[ -z "${REMOTE_DIR}" || ! "${REMOTE_DIR}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "--remote-dir may contain only letters, numbers, '.', '_', '/', and '-'." >&2
    exit 2
fi
if [[ "${REMOTE_DIR}" == ".." || "${REMOTE_DIR}" == ../* || "${REMOTE_DIR}" == */../* ]]; then
    echo "--remote-dir must not traverse through '..'." >&2
    exit 2
fi
if [[ "${RUN_NAME}" == */* || "${RUN_NAME}" == *".."* ]]; then
    echo "--name must be a simple run name without slashes or '..'." >&2
    exit 2
fi
for numeric_value in "${EPOCHS}" "${IMGSZ}" "${BATCH}" "${PATIENCE}"; do
    if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "Expected an integer, got: ${numeric_value}" >&2
        exit 2
    fi
done
if [[ "${EPOCHS}" -lt 1 || "${IMGSZ}" -lt 1 || "${BATCH}" -lt 1 ]]; then
    echo "Epochs, image size, and batch size must be positive." >&2
    exit 2
fi

for command_name in ssh rsync; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command not found: ${command_name}" >&2
        exit 1
    fi
done

remote_command() {
    local command="bash -s --"
    local argument
    local quoted
    for argument in "$@"; do
        printf -v quoted "%q" "${argument}"
        command+=" ${quoted}"
    done
    ssh "${HOST}" "${command}"
}

sync_project() {
    local remote_dir_quoted
    printf -v remote_dir_quoted "%q" "${REMOTE_DIR}"
    ssh "${HOST}" "mkdir -p ${remote_dir_quoted}"
    rsync \
        --archive \
        --compress \
        --progress \
        --exclude ".git/" \
        --exclude ".venv/" \
        --exclude "__pycache__/" \
        --exclude "*.pyc" \
        --exclude ".DS_Store" \
        --exclude "runs/" \
        --exclude "data_preprocessed/" \
        --exclude "configs/generated/" \
        "${REPO_ROOT}/" \
        "${HOST}:${REMOTE_DIR}/"
}

if [[ "${ACTION}" == "start" ]]; then
    if [[ "${SKIP_SYNC}" != "true" ]]; then
        echo "Syncing project and source data to ${HOST}:${REMOTE_DIR}/ ..."
        sync_project
    fi

    echo "Preparing the remote environment and launching training ..."
    remote_command \
        "${REMOTE_DIR}" \
        "${REMOTE_PYTHON}" \
        "${DATASET_CONFIG}" \
        "${PREPROCESS_PRESET}" \
        "${PREPROCESS_CONFIG}" \
        "${MODEL}" \
        "${EPOCHS}" \
        "${IMGSZ}" \
        "${BATCH}" \
        "${DEVICE}" \
        "${PATIENCE}" \
        "${RUN_NAME}" <<'REMOTE_START'
set -euo pipefail

remote_dir="$1"
remote_python="$2"
dataset_config="$3"
preprocess_preset="$4"
preprocess_config="$5"
model="$6"
epochs="$7"
imgsz="$8"
batch="$9"
device="${10}"
patience="${11}"
run_name="${12}"

case "${remote_dir}" in
    "~/"*) remote_dir="${HOME}/${remote_dir#~/}" ;;
esac
cd "${remote_dir}"
mkdir -p .remote_training
pid_file=".remote_training/${run_name}.pid"
log_file=".remote_training/${run_name}.log"

if [[ -f "${pid_file}" ]]; then
    old_pid="$(cat "${pid_file}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
        echo "Training '${run_name}' is already running as PID ${old_pid}." >&2
        exit 1
    fi
fi

if [[ ! -x .venv/bin/python ]]; then
    "${remote_python}" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -e .

if [[ "${device}" != "cpu" && "${device}" != "mps" ]]; then
    .venv/bin/python -c \
        'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print("GPU:", torch.cuda.get_device_name(0))'
fi

training_command=(
    .venv/bin/python scripts/train.py
    --config "${dataset_config}"
    --model "${model}"
    --epochs "${epochs}"
    --imgsz "${imgsz}"
    --batch "${batch}"
    --device "${device}"
    --patience "${patience}"
    --project runs/segment
    --name "${run_name}"
)
if [[ -n "${preprocess_preset}" ]]; then
    training_command+=(
        --preprocess-preset "${preprocess_preset}"
        --preprocess-config "${preprocess_config}"
        --rebuild-preprocessed
    )
fi

nohup "${training_command[@]}" >"${log_file}" 2>&1 </dev/null &
training_pid=$!
echo "${training_pid}" >"${pid_file}"
sleep 2
if ! kill -0 "${training_pid}" 2>/dev/null; then
    echo "Training exited during startup. Remote log:" >&2
    tail -n 100 "${log_file}" >&2
    exit 1
fi

echo "Training '${run_name}' started as PID ${training_pid}."
echo "Log: ${remote_dir}/${log_file}"
echo "Results: ${remote_dir}/runs/segment/"
REMOTE_START
    echo
    echo "Use '$0 status --host ${HOST} --remote-dir ${REMOTE_DIR} --name ${RUN_NAME}' to check it."
    echo "Use '$0 logs --host ${HOST} --remote-dir ${REMOTE_DIR} --name ${RUN_NAME}' to read the log."
    exit 0
fi

if [[ "${ACTION}" == "status" ]]; then
    remote_command "${REMOTE_DIR}" "${RUN_NAME}" <<'REMOTE_STATUS'
set -euo pipefail
remote_dir="$1"
run_name="$2"
case "${remote_dir}" in
    "~/"*) remote_dir="${HOME}/${remote_dir#~/}" ;;
esac
pid_file="${remote_dir}/.remote_training/${run_name}.pid"
if [[ ! -f "${pid_file}" ]]; then
    echo "No PID file found for '${run_name}'." >&2
    exit 1
fi
pid="$(cat "${pid_file}")"
if kill -0 "${pid}" 2>/dev/null; then
    echo "Training '${run_name}' is running as PID ${pid}."
else
    echo "Training '${run_name}' is not running (last PID: ${pid})."
fi
REMOTE_STATUS
    exit 0
fi

if [[ "${ACTION}" == "logs" ]]; then
    remote_command "${REMOTE_DIR}" "${RUN_NAME}" <<'REMOTE_LOGS'
set -euo pipefail
remote_dir="$1"
run_name="$2"
case "${remote_dir}" in
    "~/"*) remote_dir="${HOME}/${remote_dir#~/}" ;;
esac
log_file="${remote_dir}/.remote_training/${run_name}.log"
if [[ ! -f "${log_file}" ]]; then
    echo "No log found for '${run_name}'." >&2
    exit 1
fi
tail -n 100 "${log_file}"
REMOTE_LOGS
    exit 0
fi

safe_host="$(printf "%s" "${HOST}" | tr -c "A-Za-z0-9_.-" "_")"
if [[ -z "${LOCAL_RESULTS}" ]]; then
    LOCAL_RESULTS="${REPO_ROOT}/runs/remote/${safe_host}"
fi
mkdir -p "${LOCAL_RESULTS}"
echo "Downloading remote results to ${LOCAL_RESULTS}/ ..."
rsync \
    --archive \
    --compress \
    --progress \
    "${HOST}:${REMOTE_DIR}/runs/segment/" \
    "${LOCAL_RESULTS}/"
echo "Results downloaded to ${LOCAL_RESULTS}/"
