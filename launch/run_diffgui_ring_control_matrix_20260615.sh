#!/usr/bin/env bash
set -euo pipefail

# ── DiffGui ring-control matrix batch 2 (2026-06-15) ──
# Runs 5 groups sequentially:
#   09 qed                                 (single)
#   10 sa                                  (single)
#   11 ring_complexity                     (single)
#   12 le + ring_complexity + aromatic_scaffold  (composite)
#   13 docking + ring_complexity + aromatic_scaffold  (composite)
#
# Common settings: 8 GPU, per-rank batch 18, 200 steps, 10 epochs, lr=3e-5,
#                  PPO epochs 1, freeze last 3 blocks, bf16.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR="$ROOT/launch/generate_diffgui_ring_control_matrix_20260615.py"
MATRIX_DIR="$ROOT/launch/archive/generated/diffgui_ring_control_matrix_20260615"
LOG_DIR="$ROOT/launch/archive/logs/diffgui_ring_control_matrix_20260615"

# ── Overridable ──
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
START_AT="${START_AT:-1}"
STOP_AFTER="${STOP_AFTER:-999}"
SLEEP_BETWEEN="${SLEEP_BETWEEN_RUNS:-30}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"
CONDA_ENV="diffgui"   # DiffGui RL 必须用 diffgui 环境

# ── Preflight ──
mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "============================================================"
echo "[diffgui-matrix-b2] generating configs …"
conda run -n dif python "$GENERATOR"
echo "[diffgui-matrix-b2] configs  = $MATRIX_DIR"
echo "[diffgui-matrix-b2] logs     = $LOG_DIR"
echo "[diffgui-matrix-b2] env      = $CONDA_ENV"
echo "[diffgui-matrix-b2] gpus     = $CUDA_VISIBLE_DEVICES"
echo "[diffgui-matrix-b2] common   = ${NPROC_PER_NODE}GPU, bsz=18/rank, 200steps, 10ep, lr=3e-5, last3, bf16"
echo "[diffgui-matrix-b2] dry_run  = $DRY_RUN"
echo "============================================================"

# ── Run loop ──
for config in $(ls "$MATRIX_DIR"/*.yaml | sort); do
    base="$(basename "$config")"
    idx="${base%%_*}"
    # strip leading zeros
    idx_num=$((10#${idx}))
    slug="${base%.yaml}"
    slug="${slug#*_}"

    if (( idx_num < START_AT )); then continue; fi
    if (( idx_num > STOP_AFTER )); then continue; fi
    if [[ -n "$ONLY" ]]; then
        if [[ ",$ONLY," != *",$slug,"* ]]; then continue; fi
    fi

    master_port=$((MASTER_PORT_BASE + idx_num))
    run_stamp="$(date +%Y%m%d_%H%M%S)"
    output_dir="$ROOT/runs/diffgui-ring-control-20260615-${idx}-${slug}_${run_stamp}"
    log_path="$LOG_DIR/${idx}_${slug}.log"

    echo "============================================================"
    echo "[diffgui-matrix-b2] idx=$idx  slug=$slug"
    echo "[diffgui-matrix-b2] config = $config"
    echo "[diffgui-matrix-b2] output = $output_dir"
    echo "[diffgui-matrix-b2] log    = $log_path"
    echo "[diffgui-matrix-b2] port   = $master_port"
    echo "[diffgui-matrix-b2] start  = $(date '+%F %T')"
    echo "============================================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[diffgui-matrix-b2] DRY_RUN – skipping execution"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    conda run -n "$CONDA_ENV" torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$master_port" \
        ddpo/main_diffgui_docking_ddp.py \
        --config "$config" \
        --output-dir "$output_dir" \
        --log-level INFO \
        >"$log_path" 2>&1

    echo "[diffgui-matrix-b2] done  idx=$idx  slug=$slug  end=$(date '+%F %T')"

    if [[ "${SLEEP_BETWEEN}" -gt 0 ]]; then
        echo "[diffgui-matrix-b2] sleeping ${SLEEP_BETWEEN}s …"
        sleep "$SLEEP_BETWEEN"
    fi
done

echo "[diffgui-matrix-b2] all selected runs finished at $(date '+%F %T')"
