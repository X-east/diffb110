#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
GENERATOR="$ROOT/launch/generate_diffgui_ring_control_matrix_20260613.py"
MATRIX_DIR="$ROOT/launch/generated/diffgui_ring_control_matrix_20260613"
LOG_DIR="$ROOT/launch/logs/diffgui_ring_control_matrix_20260613"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
START_AT="${START_AT:-1}"
STOP_AFTER="${STOP_AFTER:-999}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-30}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29660}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[diffgui-matrix] generating configs via $GENERATOR"
"$PYTHON_BIN" "$GENERATOR"

echo "[diffgui-matrix] configs=$MATRIX_DIR"
echo "[diffgui-matrix] logs=$LOG_DIR"
echo "[diffgui-matrix] common: 8GPU, per-rank batch=18, time_step=200, epochs=10, ppo_epochs=1, lr=3e-5, freeze=last3"
echo "[diffgui-matrix] ONLY=$ONLY START_AT=$START_AT STOP_AFTER=$STOP_AFTER DRY_RUN=$DRY_RUN"

should_run_slug() {
  local slug="$1"
  if [[ -z "$ONLY" ]]; then
    return 0
  fi
  [[ ",$ONLY," == *",$slug,"* ]]
}

for config in "$MATRIX_DIR"/*.yaml; do
  base="$(basename "$config")"
  idx="${base%%_*}"
  slug="${base%.yaml}"
  slug="${slug#*_}"

  if (( 10#$idx < START_AT )); then
    continue
  fi
  if (( 10#$idx > STOP_AFTER )); then
    continue
  fi
  if ! should_run_slug "$slug"; then
    continue
  fi

  master_port=$((MASTER_PORT_BASE + 10#$idx))
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  output_dir="$ROOT/OUT/diffgui-ring-control-20260613-${idx}-${slug}_${run_stamp}"
  log_path="$LOG_DIR/${base%.yaml}.log"

  echo "============================================================"
  echo "[diffgui-matrix] run idx=$idx slug=$slug"
  echo "[diffgui-matrix] config=$config"
  echo "[diffgui-matrix] output=$output_dir"
  echo "[diffgui-matrix] log=$log_path"
  echo "[diffgui-matrix] start=$(date '+%F %T')"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  conda run -n diffgui torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$master_port" \
    ddpo/main_diffgui_docking_ddp.py \
    --config "$config" \
    --output-dir "$output_dir" \
    --log-level "$LOG_LEVEL" >"$log_path" 2>&1

  echo "[diffgui-matrix] done idx=$idx slug=$slug end=$(date '+%F %T')"
  sleep "$SLEEP_BETWEEN_RUNS"
done

echo "[diffgui-matrix] all selected runs finished at $(date '+%F %T')"
