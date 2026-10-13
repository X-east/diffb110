#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
WORKERS="${WORKERS:-20}"
CPU_SET="${CPU_SET:-8-27}"
ENV_NAME="${ENV_NAME:-dif}"
QVINA_BIN="${QVINA_BIN:-/home/md666/miniforge3/envs/diffgui/bin/qvina2}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${OUTPUT_DIR}/mplconfig"

mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}"
LOG_PATH="${OUTPUT_DIR}/launcher.log"

echo "[qvina-score-only] start=$(date -Is)" | tee -a "${LOG_PATH}"
echo "[qvina-score-only] root=${ROOT} output=${OUTPUT_DIR} workers=${WORKERS} cpu_set=${CPU_SET}" | tee -a "${LOG_PATH}"
echo "[qvina-score-only] env=${ENV_NAME} qvina=${QVINA_BIN}" | tee -a "${LOG_PATH}"

extra_args=()
if [[ -n "${LIMIT_PER_TASK:-}" ]]; then
  extra_args+=(--limit-per-task "${LIMIT_PER_TASK}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi

taskset -c "${CPU_SET}" conda run -n "${ENV_NAME}" python -u "${SCRIPT_DIR}/score_qvina_only.py" \
  --output-dir "${OUTPUT_DIR}" \
  --qvina-bin "${QVINA_BIN}" \
  --workers "${WORKERS}" \
  "${extra_args[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "[qvina-score-only] done=$(date -Is)" | tee -a "${LOG_PATH}"
