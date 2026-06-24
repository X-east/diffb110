#!/usr/bin/env bash
set -euo pipefail

# Start the 2026-06-18 DiffGui next-six matrix in the background.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/launch/archive/logs/diffgui_next6_matrix_20260618"
MASTER_LOG="$LOG_DIR/_master.log"
PID_FILE="$LOG_DIR/_master.pid"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[diffgui-next6-start] root=$ROOT"
echo "[diffgui-next6-start] master_log=$MASTER_LOG"

nohup bash "$ROOT/launch/run_diffgui_next6_matrix_20260618.sh" >"$MASTER_LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[diffgui-next6-start] pid=$pid"
echo "[diffgui-next6-start] pid_file=$PID_FILE"
echo "[diffgui-next6-start] tail -f $MASTER_LOG"
