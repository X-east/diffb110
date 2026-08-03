#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
GIT_COMMON_DIR="$(git -C "${PROJECT_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
DEFAULT_DATA_ROOT="$(dirname "${GIT_COMMON_DIR}")"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/launch/diffsbdd_offline_labels_smoke_260731.yaml}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/analysis/offline_iql_labels_multi_gpu}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
RESUME="${RESUME:-1}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
GPU_IDLE_CONFIRMATIONS="${GPU_IDLE_CONFIRMATIONS:-3}"
GPU_IDLE_CONFIRMATION_INTERVAL_SECONDS="${GPU_IDLE_CONFIRMATION_INTERVAL_SECONDS:-5}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ "${PRECHECK_ONLY}" != "0" && "${PRECHECK_ONLY}" != "1" ]]; then
  echo "PRECHECK_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${ALLOW_BUSY_GPUS}" != "0" && "${ALLOW_BUSY_GPUS}" != "1" ]]; then
  echo "ALLOW_BUSY_GPUS must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${MAX_GPU_UTILIZATION}" =~ ^[0-9]+$ ]] \
  || ((MAX_GPU_UTILIZATION < 0 || MAX_GPU_UTILIZATION > 100)); then
  echo "MAX_GPU_UTILIZATION must be an integer in [0,100]" >&2
  exit 2
fi
if ! [[ "${GPU_IDLE_CONFIRMATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_IDLE_CONFIRMATIONS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${GPU_IDLE_CONFIRMATION_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "GPU_IDLE_CONFIRMATION_INTERVAL_SECONDS must be a non-negative integer" >&2
  exit 2
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
else
  mapfile -t GPU_ARRAY < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
  )
fi
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPU IDs available; set GPU_IDS explicitly" >&2
  exit 2
fi
NUM_WORKERS="${NUM_WORKERS:-${#GPU_ARRAY[@]}}"
if [[ "${NUM_WORKERS}" -le 0 || "${NUM_WORKERS}" -gt "${#GPU_ARRAY[@]}" ]]; then
  echo "NUM_WORKERS must be in [1, ${#GPU_ARRAY[@]}]" >&2
  exit 2
fi

check_selected_gpus_idle_once() {
  local phase="$1"
  local busy=0
  local worker_id gpu_id utilization compute_apps
  for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
    gpu_id="${GPU_ARRAY[worker_id]}"
    utilization="$(
      nvidia-smi \
        --id="${gpu_id}" \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits
    )"
    utilization="${utilization//[[:space:]]/}"
    compute_apps="$(
      nvidia-smi \
        --id="${gpu_id}" \
        --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits
    )"
    if ((utilization > MAX_GPU_UTILIZATION)) || [[ -n "${compute_apps}" ]]; then
      echo "GPU busy phase=${phase} gpu=${gpu_id} utilization=${utilization}% apps=${compute_apps:-none}" >&2
      busy=1
    fi
  done
  return "${busy}"
}

require_selected_gpus_idle() {
  local phase="$1"
  local confirmation
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    echo "GPU idle guard bypassed phase=${phase} ALLOW_BUSY_GPUS=1" >&2
    return
  fi
  for ((confirmation = 1; confirmation <= GPU_IDLE_CONFIRMATIONS; confirmation++)); do
    if ! check_selected_gpus_idle_once "${phase}:${confirmation}/${GPU_IDLE_CONFIRMATIONS}"; then
      echo "Refusing label launch because selected GPUs are busy" >&2
      exit 3
    fi
    if ((confirmation < GPU_IDLE_CONFIRMATIONS)); then
      sleep "${GPU_IDLE_CONFIRMATION_INTERVAL_SECONDS}"
    fi
  done
  echo "GPU idle guard passed phase=${phase} confirmations=${GPU_IDLE_CONFIRMATIONS}"
}

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/config"
snapshot_file() {
  local source_path="$1"
  local snapshot_path="$2"
  if [[ -f "${snapshot_path}" ]]; then
    if ! cmp -s "${source_path}" "${snapshot_path}"; then
      echo "Run-local script snapshot differs: ${snapshot_path}" >&2
      exit 2
    fi
    return
  fi
  cp "${source_path}" "${snapshot_path}"
}
snapshot_file "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/config/launcher_snapshot.sh"
snapshot_file \
  "${PROJECT_ROOT}/scripts/offline_rl/label_trajectory_shards.py" \
  "${OUTPUT_DIR}/config/label_trajectory_shards_snapshot.py"

echo "$(date --iso-8601=seconds) label launch preflight start"
echo "config=${CONFIG_PATH}"
echo "data_root=${DATA_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "workers=${NUM_WORKERS} gpu_ids=${GPU_ARRAY[*]}"
echo "pythonpath=${PYTHONPATH}"
echo "git_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
git -C "${PROJECT_ROOT}" status --short
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

COMMON_ARGS=(
  --config "${CONFIG_PATH}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --num-workers "${NUM_WORKERS}"
)

if [[ "${PRECHECK_ONLY}" == "0" ]]; then
  require_selected_gpus_idle "before_preflight"
fi

for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  conda run -n dif python \
    "${PROJECT_ROOT}/scripts/offline_rl/label_trajectory_shards.py" \
    "${COMMON_ARGS[@]}" \
    --worker-id "${worker_id}" \
    --gpu-id "${GPU_ARRAY[worker_id]}" \
    --preflight-only
done
echo "$(date --iso-8601=seconds) label launch preflight completed"
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  exit 0
fi

require_selected_gpus_idle "after_preflight"

declare -a WORKER_PIDS=()
cleanup_workers() {
  if [[ "${#WORKER_PIDS[@]}" -gt 0 ]]; then
    kill "${WORKER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_workers INT TERM

for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  worker_args=(
    "${COMMON_ARGS[@]}"
    --worker-id "${worker_id}"
    --gpu-id "${GPU_ARRAY[worker_id]}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    worker_args+=(--resume)
  fi
  conda run -n dif python \
    "${PROJECT_ROOT}/scripts/offline_rl/label_trajectory_shards.py" \
    "${worker_args[@]}" \
    > "${OUTPUT_DIR}/logs/worker_${worker_id}_launcher.log" 2>&1 &
  worker_pid="$!"
  WORKER_PIDS+=("${worker_pid}")
  echo "worker=${worker_id} gpu=${GPU_ARRAY[worker_id]} pid=${worker_pid} started"
done

failed=0
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  if wait "${WORKER_PIDS[worker_id]}"; then
    echo "worker=${worker_id} completed"
  else
    echo "worker=${worker_id} failed; see ${OUTPUT_DIR}/logs/worker_${worker_id}_launcher.log" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

conda run -n dif python \
  "${PROJECT_ROOT}/scripts/offline_rl/label_trajectory_shards.py" \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --worker-id 0 \
  --num-workers 1 \
  --finalize-only
echo "$(date --iso-8601=seconds) label launch completed"
