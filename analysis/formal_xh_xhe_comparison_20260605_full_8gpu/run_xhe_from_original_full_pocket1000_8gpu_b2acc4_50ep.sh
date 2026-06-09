#!/usr/bin/env bash
set -euo pipefail

CONFIG="analysis/formal_xh_xhe_comparison_20260605_full_8gpu/diffsbdd_xhe_from_original_full_pocket1000_8gpu_b2acc4_50ep.yaml"
RUN_DIR="analysis/formal_xh_xhe_comparison_20260605_full_8gpu/lightning/diffsbdd-xhe-from-original-full-pocket1000-8gpu-260605-b2acc4-50ep"
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"

RESUME_ARGS=()
if [[ "${RESUME_FROM_LAST:-0}" == "1" && -f "${LAST_CKPT}" ]]; then
  RESUME_ARGS=(--resume "${LAST_CKPT}")
  echo "[xhe-formal-pocket1000-b2acc4] Resuming from ${LAST_CKPT}"
else
  echo "[xhe-formal-pocket1000-b2acc4] Fresh start from DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

/usr/bin/time -p conda run --no-capture-output -n dif \
  python DiffSBDD/train.py \
  --config "${CONFIG}" \
  "${RESUME_ARGS[@]}"
