#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_deppa_full_continue_from85_smoke_20260911.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29868}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/OUT/diffgui-deppa-full-continue-from85-smoke_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-$ROOT/launch/archive/logs/diffgui_deppa_full_20260626_continue}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/smoke_continue_from85_${RUN_STAMP}.log}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs" "${LOG_DIR}"
cd "${ROOT}"

echo "[SMOKE] config=${CONFIG_PATH}"
echo "[SMOKE] output=${OUTPUT_DIR}"
echo "[SMOKE] log=${LOG_PATH}"
echo "[SMOKE] purpose=verify SDF-merge exception patch (no crash on bad chiral center)"
echo "[SMOKE] resume_from=epoch0085, epochs=1, epoch_offset=85 -> global epoch 86"
echo "[SMOKE] gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "[SMOKE] start=$(date '+%F %T')"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  >"${LOG_PATH}" 2>&1

echo "[SMOKE] done=$(date '+%F %T')"
echo "[SMOKE] check: grep -iE 'skipped|Traceback|AttributeError' ${LOG_PATH}"
echo "[SMOKE] expect: epoch_0086 merged files present, no AttributeError"
