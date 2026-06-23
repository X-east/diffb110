#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_deppa_last3_lr1e5_b18x8_t200_100ep_20260623.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29863}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/OUT/diffgui-deppa-last3-lr1e5-b18x8-t200-100ep_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT/launch/archive/logs/diffgui_deppa_20260623}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/diffgui_deppa_last3_lr1e5_b18x8_t200_100ep_${RUN_STAMP}.log}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs" "${LOG_DIR}"
cd "${ROOT}"

echo "[DiffGui-DDP] config=${CONFIG_PATH}"
echo "[DiffGui-DDP] output=${OUTPUT_DIR}"
echo "[DiffGui-DDP] log=${LOG_PATH}"
echo "[DiffGui-DDP] env=${CONDA_ENV}"
echo "[DiffGui-DDP] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-DDP] nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
echo "[DiffGui-DDP] common=8GPU, batch/rank=18, epochs=100, steps=200, last_n_blocks=3, lr=1e-5, weight_decay=1e-4"
echo "[DiffGui-DDP] start=$(date '+%F %T')"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  >"${LOG_PATH}" 2>&1

echo "[DiffGui-DDP] done=$(date '+%F %T')"
