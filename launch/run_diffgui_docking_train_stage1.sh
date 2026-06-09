#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-launch/diffgui_docking_train_stage1_last1_b16.yaml}"
DEVICE="${DEVICE:-cuda:7}"
DOCKING_GPU_ID="${DOCKING_GPU_ID:-7}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-OUT/diffgui-docking-stage1-last1-b16_${RUN_STAMP}}"

mkdir -p "${OUTPUT_DIR}/launch"

echo "[DiffGui stage1] config=${CONFIG_PATH}"
echo "[DiffGui stage1] output=${OUTPUT_DIR}"
echo "[DiffGui stage1] device=${DEVICE} docking_gpu=${DOCKING_GPU_ID}"

conda run -n diffgui python ddpo/main_diffgui_docking_train.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --docking-gpu-id "${DOCKING_GPU_ID}" \
  --log-level "${LOG_LEVEL}" \
  2>&1 | tee "${OUTPUT_DIR}/launch/console.log"
