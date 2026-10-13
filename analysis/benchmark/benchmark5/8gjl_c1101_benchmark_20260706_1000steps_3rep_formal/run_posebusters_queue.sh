#!/usr/bin/env bash
# Low-load PoseBusters queue for the formal five-model benchmark.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${BASE}/posebusters_queue}"
MAX_LOAD="${MAX_LOAD:-20}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-60}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUT_DIR}/logs"
LOG="${OUT_DIR}/logs/posebusters_queue_${RUN_TAG}.log"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

args=(
  "${BASE}/posebusters_queue.py"
  --out-dir "${OUT_DIR}"
  --max-load "${MAX_LOAD}"
  --check-interval "${CHECK_INTERVAL}"
  --sleep-between "${SLEEP_BETWEEN}"
)

if [[ -n "${LIMIT}" ]]; then
  args+=(--limit "${LIMIT}")
fi

echo "[posebusters-queue] started at $(date -Is)" | tee -a "${LOG}"
echo "[posebusters-queue] out_dir=${OUT_DIR}" | tee -a "${LOG}"
echo "[posebusters-queue] max_load=${MAX_LOAD} check_interval=${CHECK_INTERVAL} sleep_between=${SLEEP_BETWEEN}" | tee -a "${LOG}"
echo "[posebusters-queue] threads OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}" | tee -a "${LOG}"

nice -n 15 ionice -c2 -n7 conda run --no-capture-output -n dif python "${args[@]}" 2>&1 | tee -a "${LOG}"

echo "[posebusters-queue] completed at $(date -Is)" | tee -a "${LOG}"
