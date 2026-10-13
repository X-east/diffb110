#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_original_t200_8gpu_trajectory_smoke_20260727.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29727}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui_collection/diffgui-original-t200-8gpu-trajectory-smoke_${RUN_STAMP}}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/logs/launch.log}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs"
cd "${ROOT}"

echo "[DiffGui-Collect-Smoke] config=${CONFIG_PATH}"
echo "[DiffGui-Collect-Smoke] output=${OUTPUT_DIR}"
echo "[DiffGui-Collect-Smoke] env=${CONDA_ENV} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-Collect-Smoke] 8 ranks x 10 attempts, 200 steps, frozen original checkpoint"

conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_collect_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  >"${LOG_PATH}" 2>&1

echo "[DiffGui-Collect-Smoke] completed output=${OUTPUT_DIR}"
