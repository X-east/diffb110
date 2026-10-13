#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="${PROJECT_ROOT}/launch/run_diffsbdd_offline_labels_multi_gpu.sh"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/launch/diffsbdd_offline_labels_formal_v1_260803.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/disk6/DiffB110}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/analysis/offline_iql_labels_formal_v1_guarded_260803}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 2
fi
bash -n "${BASH_SOURCE[0]}"
bash -n "${LAUNCHER}"

mkdir -p "${OUTPUT_DIR}/config" "${OUTPUT_DIR}/logs"
WAITER_SNAPSHOT="${OUTPUT_DIR}/config/waiter_snapshot.sh"
if [[ -f "${WAITER_SNAPSHOT}" ]]; then
  if ! cmp -s "${BASH_SOURCE[0]}" "${WAITER_SNAPSHOT}"; then
    echo "Run-local waiter snapshot differs: ${WAITER_SNAPSHOT}" >&2
    exit 2
  fi
else
  cp "${BASH_SOURCE[0]}" "${WAITER_SNAPSHOT}"
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "$(date --iso-8601=seconds) formal label waiter attempt=${attempt}"
  set +e
  CONFIG_PATH="${CONFIG_PATH}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  GPU_IDS="${GPU_IDS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PRECHECK_ONLY=0 \
  RESUME=1 \
  ALLOW_BUSY_GPUS=0 \
    bash "${LAUNCHER}"
  exit_code="$?"
  set -e
  if [[ "${exit_code}" == "0" ]]; then
    echo "$(date --iso-8601=seconds) formal label waiter completed"
    exit 0
  fi
  if [[ "${exit_code}" != "3" ]]; then
    echo "$(date --iso-8601=seconds) formal label waiter failed exit_code=${exit_code}" >&2
    exit "${exit_code}"
  fi
  echo "$(date --iso-8601=seconds) GPUs busy; retry in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
