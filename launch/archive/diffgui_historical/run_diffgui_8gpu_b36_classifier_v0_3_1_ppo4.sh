#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-launch/diffgui_8gpu_b36_classifier_v0_3_1_200step_ppo4.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29643}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-OUT/diffgui-8gpu-b36-classifier-v0-3-1-ppo4_${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================"
echo "[DiffGui-8GPU] config=${CONFIG_PATH}"
echo "[DiffGui-8GPU] output=${OUTPUT_DIR}"
echo "[DiffGui-8GPU] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-8GPU] nproc_per_node=${NPROC_PER_NODE}"
echo "[DiffGui-8GPU] num_timesteps=200"
echo "[DiffGui-8GPU] per_rank_batch=36  global_batch=$((NPROC_PER_NODE * 36))"
echo "[DiffGui-8GPU] ppo_epochs=4  total_epochs=100"
echo "[DiffGui-8GPU] rewards=docking(0.5)+chembl_classifier_v0.3.1(0.5)"
echo "[DiffGui-8GPU] classifier=chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt"
echo "[DiffGui-8GPU] reward_env=dif  train_env=diffgui"
echo "============================================"

conda run -n diffgui torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level "${LOG_LEVEL}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/launch.log"
