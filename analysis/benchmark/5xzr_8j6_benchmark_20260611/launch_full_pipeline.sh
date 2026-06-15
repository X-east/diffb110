#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL="${TOTAL:-500}"
TIMESTEPS="${TIMESTEPS:-500}"
GPUS="${GPUS:-0 1 2 3 4 5 6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PMDM_NUM_ATOM="${PMDM_NUM_ATOM:-28}"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/summary" "${SCRIPT_DIR}/mplconfig/pipeline"
export MPLCONFIGDIR="${SCRIPT_DIR}/mplconfig/pipeline"

echo "[5xzr_full] started at $(date -Is)"
echo "[5xzr_full] total=${TOTAL} timesteps=${TIMESTEPS} gpus=${GPUS} batch_size=${BATCH_SIZE} pmdm_num_atom=${PMDM_NUM_ATOM}"

TOTAL="${TOTAL}" TIMESTEPS="${TIMESTEPS}" GPUS="${GPUS}" \
  bash "${SCRIPT_DIR}/launch_core_benchmark.sh"

TOTAL="${TOTAL}" TIMESTEPS="${TIMESTEPS}" GPUS="${GPUS}" PMDM_NUM_ATOM="${PMDM_NUM_ATOM}" \
  bash "${SCRIPT_DIR}/launch_second_benchmark.sh"

BATCH_SIZE="${BATCH_SIZE}" \
  bash "${SCRIPT_DIR}/launch_docking_six_models.sh"

conda run -n dif python "${SCRIPT_DIR}/summarize_benchmark.py" \
  --run-root "${SCRIPT_DIR}" \
  --out-dir "${SCRIPT_DIR}/summary" \
  --label "5XZR/8J6 formal benchmark" \
  --file-prefix benchmark

echo "[5xzr_full] finished at $(date -Is)"
