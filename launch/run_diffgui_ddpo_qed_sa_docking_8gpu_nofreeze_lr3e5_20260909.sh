#!/usr/bin/env bash
set -euo pipefail

# DiffGui 标准 DDPO：QED 0.25 + SA 0.25 + Docking 0.50
# 修正上一轮（last2 + lr1e-5 几乎没更新）：freeze=none + lr=3e-5
# 入口: ddpo/main_diffgui_docking_ddp.py
# 自动止血线由 watchdog_pb_bondlength.py 看护（键长<0.80 自动停训）

CONFIG_PATH="${CONFIG_PATH:-launch/diffgui_ddpo_qed_sa_docking_8gpu_b8_t200_100ep_nofreeze_lr3e5_20260909.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29643}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-OUT/diffgui-ddpo-qed-sa-docking-b8-t200-100ep-nofreeze-lr3e5_${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================"
echo "[DiffGui-DDPO] QED+SA+Docking  freeze=none lr=3e-5"
echo "[DiffGui-DDPO] config=${CONFIG_PATH}"
echo "[DiffGui-DDPO] output=${OUTPUT_DIR}"
echo "[DiffGui-DDPO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-DDPO] nproc_per_node=${NPROC_PER_NODE}"
echo "[DiffGui-DDPO] per_rank_batch=8  global_batch=$((NPROC_PER_NODE * 8))"
echo "[DiffGui-DDPO] reward=qed_0.25+sa_0.25+dock_0.5  lr=3e-5  freeze=none  T=200"
echo "============================================"

conda run -n diffgui torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/launch.log"
