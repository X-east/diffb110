#!/usr/bin/env bash
set -euo pipefail

# DiffGui 标准 DDPO（非 DePPA）：QED + SA 等权简单奖励
# 入口: ddpo/main_diffgui_docking_ddp.py
# 环境: diffgui 专用 conda 环境（与 dif 隔离，规则要求）

CONFIG_PATH="${CONFIG_PATH:-launch/diffgui_ddpo_qed_sa_8gpu_b16_lr1e5_last2_20260908.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29641}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-OUT/diffgui-ddpo-qed-sa-b16-lr1e5-last2_${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================"
echo "[DiffGui-DDPO] QED+SA simple reward (no docking, no DePPA)"
echo "[DiffGui-DDPO] config=${CONFIG_PATH}"
echo "[DiffGui-DDPO] output=${OUTPUT_DIR}"
echo "[DiffGui-DDPO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-DDPO] nproc_per_node=${NPROC_PER_NODE}"
echo "[DiffGui-DDPO] per_rank_batch=16  global_batch=$((NPROC_PER_NODE * 16))"
echo "[DiffGui-DDPO] reward=qed_0.5+sa_0.5  lr=1e-5  freeze=last_2_blocks  T=200"
echo "============================================"

conda run -n diffgui torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/launch.log"
