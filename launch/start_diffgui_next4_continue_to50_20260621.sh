#!/usr/bin/env bash
set -euo pipefail

# Start DG-NEXT-04 continue-to-50 training in the background.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/launch/archive/logs/diffgui_continue_to50_20260621"
MASTER_LOG="$LOG_DIR/_master.log"
PID_FILE="$LOG_DIR/_master.pid"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[diffgui-next4-continue-to50-start] root=$ROOT"
echo "[diffgui-next4-continue-to50-start] master_log=$MASTER_LOG"

nohup bash "$ROOT/launch/run_diffgui_next4_continue_to50_20260621.sh" >"$MASTER_LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[diffgui-next4-continue-to50-start] pid=$pid"
echo "[diffgui-next4-continue-to50-start] pid_file=$PID_FILE"
echo "[diffgui-next4-continue-to50-start] tail -f $MASTER_LOG"
