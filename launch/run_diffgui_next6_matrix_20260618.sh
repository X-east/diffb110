#!/usr/bin/env bash
set -euo pipefail

# DiffGui next-six matrix, 2026-06-18.
# Runs six 20-epoch DiffGui PPO/DDPO reward tests sequentially.
# Common settings: 8 GPU, per-rank batch 18, 200 diffusion steps, lr=3e-5,
# PPO epochs 1, freeze last 3 blocks, bf16.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR="$ROOT/launch/generate_diffgui_next6_matrix_20260618.py"
MATRIX_DIR="$ROOT/launch/archive/generated/diffgui_next6_matrix_20260618"
LOG_DIR="$ROOT/launch/archive/logs/diffgui_next6_matrix_20260618"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
START_AT="${START_AT:-1}"
STOP_AFTER="${STOP_AFTER:-999}"
SLEEP_BETWEEN="${SLEEP_BETWEEN_RUNS:-30}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29820}"
CONDA_ENV="${CONDA_ENV:-diffgui}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "============================================================"
echo "[diffgui-next6] generating configs"
conda run -n dif python "$GENERATOR"
echo "[diffgui-next6] configs   = $MATRIX_DIR"
echo "[diffgui-next6] logs      = $LOG_DIR"
echo "[diffgui-next6] env       = $CONDA_ENV"
echo "[diffgui-next6] gpus      = $CUDA_VISIBLE_DEVICES"
echo "[diffgui-next6] common    = ${NPROC_PER_NODE}GPU, batch/rank=18, epochs=20, steps=200"
echo "[diffgui-next6] dry_run   = $DRY_RUN"
echo "[diffgui-next6] start_at  = $START_AT"
echo "[diffgui-next6] stop_after= $STOP_AFTER"
echo "[diffgui-next6] only      = ${ONLY:-<all>}"
echo "============================================================"

for config in $(find "$MATRIX_DIR" -maxdepth 1 -type f -name "*.yaml" | sort); do
    base="$(basename "$config")"
    idx="${base%%_*}"
    idx_num=$((10#${idx}))
    slug="${base%.yaml}"
    slug="${slug#*_}"

    if (( idx_num < START_AT )); then
        continue
    fi
    if (( idx_num > STOP_AFTER )); then
        continue
    fi
    if [[ -n "$ONLY" && ",$ONLY," != *",$slug,"* ]]; then
        continue
    fi

    master_port=$((MASTER_PORT_BASE + idx_num))
    run_stamp="$(date +%Y%m%d_%H%M%S)"
    output_dir="$ROOT/OUT/diffgui-next6-20260618-${idx}-${slug}_${run_stamp}"
    log_path="$LOG_DIR/${idx}_${slug}.log"

    echo "============================================================"
    echo "[diffgui-next6] idx=$idx slug=$slug"
    echo "[diffgui-next6] config = $config"
    echo "[diffgui-next6] output = $output_dir"
    echo "[diffgui-next6] log    = $log_path"
    echo "[diffgui-next6] port   = $master_port"
    echo "[diffgui-next6] start  = $(date '+%F %T')"
    echo "============================================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[diffgui-next6] DRY_RUN: skipped execution"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    conda run -n "$CONDA_ENV" torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$master_port" \
        ddpo/main_diffgui_docking_ddp.py \
        --config "$config" \
        --output-dir "$output_dir" \
        --epochs 20 \
        --batch-size 18 \
        --log-level INFO \
        >"$log_path" 2>&1

    echo "[diffgui-next6] done idx=$idx slug=$slug end=$(date '+%F %T')"

    if [[ "$SLEEP_BETWEEN" -gt 0 ]]; then
        echo "[diffgui-next6] sleeping ${SLEEP_BETWEEN}s"
        sleep "$SLEEP_BETWEEN"
    fi
done

echo "[diffgui-next6] all selected runs finished at $(date '+%F %T')"
