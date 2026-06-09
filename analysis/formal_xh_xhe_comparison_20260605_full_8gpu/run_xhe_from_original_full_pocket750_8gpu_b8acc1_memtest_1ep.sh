#!/usr/bin/env bash
set -euo pipefail

CONFIG="analysis/formal_xh_xhe_comparison_20260605_full_8gpu/diffsbdd_xhe_from_original_full_pocket750_8gpu_b8acc1_memtest_1ep.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

/usr/bin/time -p conda run --no-capture-output -n dif \
  python DiffSBDD/train.py \
  --config "${CONFIG}"
