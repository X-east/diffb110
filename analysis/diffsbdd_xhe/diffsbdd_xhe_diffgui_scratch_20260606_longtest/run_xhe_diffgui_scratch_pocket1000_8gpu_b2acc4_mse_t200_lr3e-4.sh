#!/usr/bin/env bash
set -euo pipefail

CONFIG="analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/diffsbdd_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_mse_t200_lr3e-4.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[xhe-diffgui-scratch-mse-t200-lr3e-4] Starting from scratch with config ${CONFIG}"
/usr/bin/time -p conda run --no-capture-output -n dif \
  python DiffSBDD/train.py \
  --config "${CONFIG}"
