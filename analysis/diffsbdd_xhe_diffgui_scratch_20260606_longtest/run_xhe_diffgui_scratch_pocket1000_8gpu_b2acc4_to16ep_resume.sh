#!/usr/bin/env bash
set -euo pipefail

CONFIG="analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/diffsbdd_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_to16ep_resume.yaml"
RUN_DIR="analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/lightning/diffsbdd-xhe-diffgui-scratch-pocket1000-8gpu-260606-b2acc4-8ep"
RESUME_CKPT="${RUN_DIR}/checkpoints/last_continue_to16.ckpt"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "[xhe-diffgui-scratch-continue] Missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[xhe-diffgui-scratch-continue] Resuming to max_epochs=16 from ${RESUME_CKPT}"
/usr/bin/time -p conda run --no-capture-output -n dif \
  python DiffSBDD/train.py \
  --config "${CONFIG}" \
  --resume "${RESUME_CKPT}"
