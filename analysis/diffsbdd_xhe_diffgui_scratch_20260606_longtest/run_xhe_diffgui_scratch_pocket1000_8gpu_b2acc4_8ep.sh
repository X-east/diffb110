#!/usr/bin/env bash
set -euo pipefail

CONFIG="analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/diffsbdd_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.yaml"
RUN_DIR="analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/lightning/diffsbdd-xhe-diffgui-scratch-pocket1000-8gpu-260606-b2acc4-8ep"
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"

RESUME_ARGS=()
if [[ "${RESUME_FROM_LAST:-0}" == "1" && -f "${LAST_CKPT}" ]]; then
  RESUME_ARGS=(--resume "${LAST_CKPT}")
  echo "[xhe-diffgui-scratch-longtest] Resuming from ${LAST_CKPT}"
else
  echo "[xhe-diffgui-scratch-longtest] Fresh scratch start"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

/usr/bin/time -p conda run --no-capture-output -n dif \
  python DiffSBDD/train.py \
  --config "${CONFIG}" \
  "${RESUME_ARGS[@]}"
