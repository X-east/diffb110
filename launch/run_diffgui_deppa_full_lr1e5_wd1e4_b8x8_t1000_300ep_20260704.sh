#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_deppa_full_lr1e5_wd1e4_b8x8_t1000_300ep_20260704.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29874}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t1000-300ep_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT/launch/archive/logs/diffgui_deppa_full_t1000_20260704}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/diffgui_deppa_full_lr1e5_wd1e4_b8x8_t1000_300ep_${RUN_STAMP}.log}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs" "${LOG_DIR}"
cd "${ROOT}"

echo "[DiffGui-DDP-Full-T1000] config=${CONFIG_PATH}"
echo "[DiffGui-DDP-Full-T1000] output=${OUTPUT_DIR}"
echo "[DiffGui-DDP-Full-T1000] log=${LOG_PATH}"
echo "[DiffGui-DDP-Full-T1000] env=${CONDA_ENV}"
echo "[DiffGui-DDP-Full-T1000] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-DDP-Full-T1000] nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
echo "[DiffGui-DDP-Full-T1000] common=8GPU, batch/rank=8, global_batch=64, epochs=300, checkpoint_every=1, steps=1000, seed_mode=gpu_id, freeze=none, lr=1e-5, weight_decay=1e-4, ppo_epochs=2, target=8GJL:C1101"
echo "[DiffGui-DDP-Full-T1000] start=$(date '+%F %T')"

EXTRA_ARGS=()
if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
  EXTRA_ARGS+=(--epochs "${EPOCHS_OVERRIDE}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  "${EXTRA_ARGS[@]}" \
  >"${LOG_PATH}" 2>&1

echo "[DiffGui-DDP-Full-T1000] done=$(date '+%F %T')"
