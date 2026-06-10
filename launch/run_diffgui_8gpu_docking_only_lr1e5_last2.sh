#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-launch/diffgui_8gpu_b24_docking_only_lr1e5_last2.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29632}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-OUT/diffgui-8gpu-docking-only-lr1e5-last2_${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================"
echo "[DiffGui-8GPU] pure docking trend test"
echo "[DiffGui-8GPU] config=${CONFIG_PATH}"
echo "[DiffGui-8GPU] output=${OUTPUT_DIR}"
echo "[DiffGui-8GPU] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-8GPU] nproc_per_node=${NPROC_PER_NODE}"
echo "[DiffGui-8GPU] per_rank_batch=24  global_batch=$((NPROC_PER_NODE * 24))"
echo "[DiffGui-8GPU] reward=docking_only"
echo "[DiffGui-8GPU] lr=1e-5  freeze=last_2_blocks"
echo "[DiffGui-8GPU] reward_env=dif  unidock=GPU(per-rank)"
echo "============================================"

conda run -n diffgui torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/launch.log"
