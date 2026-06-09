#!/usr/bin/env bash
set -euo pipefail

CONFIG="analysis/formal_xh_xhe_comparison_20260605_full_8gpu/diffsbdd_xhe_auxwarm_full_8gpu_b1acc4_20ep.yaml"
RUN_DIR="analysis/formal_xh_xhe_comparison_20260605_full_8gpu/lightning/diffsbdd-xhe-auxwarm-full-8gpu-260605-b1acc4-20ep"
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"

RESUME_ARGS=()
if [[ -f "${LAST_CKPT}" ]]; then
  RESUME_ARGS=(--resume "${LAST_CKPT}")
  echo "[xhe-formal] Resuming from ${LAST_CKPT}"
else
  echo "[xhe-formal] Starting from config warm_start_checkpoint"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

/usr/bin/time -p conda run --no-capture-output -n dif \
  python DiffSBDD/train.py \
  --config "${CONFIG}" \
  "${RESUME_ARGS[@]}"
