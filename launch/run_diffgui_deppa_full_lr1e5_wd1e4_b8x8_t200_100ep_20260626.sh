#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_deppa_full_lr1e5_wd1e4_b8x8_t200_100ep_20260626.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29866}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT/launch/archive/logs/diffgui_deppa_full_20260626}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/diffgui_deppa_full_lr1e5_wd1e4_b8x8_t200_100ep_${RUN_STAMP}.log}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs" "${LOG_DIR}"
cd "${ROOT}"

echo "[DiffGui-DDP-Full] config=${CONFIG_PATH}"
echo "[DiffGui-DDP-Full] output=${OUTPUT_DIR}"
echo "[DiffGui-DDP-Full] log=${LOG_PATH}"
echo "[DiffGui-DDP-Full] env=${CONDA_ENV}"
echo "[DiffGui-DDP-Full] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-DDP-Full] nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
echo "[DiffGui-DDP-Full] common=8GPU, batch/rank=8, global_batch=64, epochs=100, steps=200, freeze=none, lr=1e-5, weight_decay=1e-4, ppo_epochs=2, target=8GJL:C1101"
echo "[DiffGui-DDP-Full] start=$(date '+%F %T')"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  >"${LOG_PATH}" 2>&1

echo "[DiffGui-DDP-Full] done=$(date '+%F %T')"
