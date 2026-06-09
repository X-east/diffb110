#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-launch/diffgui_docking_train_ddp_smoke.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29621}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-OUT/diffgui-docking-ddp-smoke_${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs"

echo "[DiffGui-DDP] config=${CONFIG_PATH}"
echo "[DiffGui-DDP] output=${OUTPUT_DIR}"
echo "[DiffGui-DDP] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-DDP] nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"

conda run -n diffgui torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/launch.log"
