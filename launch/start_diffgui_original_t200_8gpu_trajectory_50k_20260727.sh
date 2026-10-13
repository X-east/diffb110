#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui_collection/diffgui-original-t200-8gpu-trajectory-50k_${RUN_STAMP}}"
WRAPPER_LOG="${WRAPPER_LOG:-$OUTPUT_DIR/logs/launcher_stdout.log}"
PID_PATH="${PID_PATH:-$OUTPUT_DIR/launcher.pid}"

mkdir -p "${OUTPUT_DIR}/logs"
cd "${ROOT}"

nohup env \
  RUN_STAMP="${RUN_STAMP}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_PATH="${OUTPUT_DIR}/logs/launch.log" \
  bash launch/run_diffgui_original_t200_8gpu_trajectory_50k_20260727.sh \
  >"${WRAPPER_LOG}" 2>&1 &

LAUNCHER_PID=$!
echo "${LAUNCHER_PID}" >"${PID_PATH}"
echo "[DiffGui-Collect-50k] launcher_pid=${LAUNCHER_PID}"
echo "[DiffGui-Collect-50k] output=${OUTPUT_DIR}"
echo "[DiffGui-Collect-50k] wrapper_log=${WRAPPER_LOG}"

