#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
GENERATOR="$ROOT/launch/generate_diffsbdd_single_reward_matrix_20260612.py"
MATRIX_DIR="$ROOT/launch/generated/diffsbdd_single_reward_matrix_20260612"
LOG_DIR="$ROOT/launch/logs/diffsbdd_single_reward_matrix_20260612"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
START_AT="${START_AT:-1}"
STOP_AFTER="${STOP_AFTER:-999}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-30}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[matrix] generating configs via $GENERATOR"
"$PYTHON_BIN" "$GENERATOR"

echo "[matrix] configs=$MATRIX_DIR"
echo "[matrix] logs=$LOG_DIR"
echo "[matrix] common: 8GPU, per-rank batch=18, time_step=200, ppo_epochs=2, epochs=10, lr=2e-5, kl=0.05"
echo "[matrix] ONLY=$ONLY START_AT=$START_AT STOP_AFTER=$STOP_AFTER DRY_RUN=$DRY_RUN"

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

  log_path="$LOG_DIR/${base%.yaml}.log"
  echo "============================================================"
  echo "[matrix] run idx=$idx slug=$slug"
  echo "[matrix] config=$config"
  echo "[matrix] log=$log_path"
  echo "[matrix] start=$(date '+%F %T')"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  CONFIG_PATH="$config" LOG_LEVEL="$LOG_LEVEL" bash "$ROOT/launch/run.sh" >"$log_path" 2>&1

  echo "[matrix] done idx=$idx slug=$slug end=$(date '+%F %T')"
  sleep "$SLEEP_BETWEEN_RUNS"
done

echo "[matrix] all selected runs finished at $(date '+%F %T')"
