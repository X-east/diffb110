#!/usr/bin/env bash
set -euo pipefail

# Continue DG-NEXT-04 from epoch 20 to total epoch 50.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/launch/diffgui_next4_continue_to50_20260621.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT/launch/archive/logs/diffgui_continue_to50_20260621}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29854}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui-next6-20260618-04-ring_last3_pathology_guard-continue-to50_${RUN_STAMP}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/04_ring_last3_pathology_guard_continue_to50_${RUN_STAMP}.log}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "============================================================"
echo "[diffgui-next4-continue-to50] config = $CONFIG"
echo "[diffgui-next4-continue-to50] output = $OUTPUT_DIR"
echo "[diffgui-next4-continue-to50] log    = $LOG_PATH"
echo "[diffgui-next4-continue-to50] env    = $CONDA_ENV"
echo "[diffgui-next4-continue-to50] gpus   = $CUDA_VISIBLE_DEVICES"
echo "[diffgui-next4-continue-to50] common = ${NPROC_PER_NODE}GPU, batch/rank=18, epochs=30, epoch_offset=20, steps=200"
echo "[diffgui-next4-continue-to50] port   = $MASTER_PORT"
echo "[diffgui-next4-continue-to50] start  = $(date '+%F %T')"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
conda run -n "$CONDA_ENV" torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    ddpo/main_diffgui_docking_ddp.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 30 \
    --batch-size 18 \
    --log-level INFO \
    >"$LOG_PATH" 2>&1

echo "[diffgui-next4-continue-to50] done end=$(date '+%F %T')"
