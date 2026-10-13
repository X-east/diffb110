#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui_collection/diffgui-original-t200-8gpu-trajectory-50k_20260727_212433}"
WRAPPER_LOG="${WRAPPER_LOG:-$OUTPUT_DIR/logs/resume_launcher_stdout_20260728.log}"
PID_PATH="${PID_PATH:-$OUTPUT_DIR/resume_launcher_20260728.pid}"

mkdir -p "${OUTPUT_DIR}/logs"
cd "${ROOT}"

nohup env \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash launch/run_diffgui_original_t200_8gpu_trajectory_50k_resume_20260728.sh \
  >"${WRAPPER_LOG}" 2>&1 &

LAUNCHER_PID=$!
echo "${LAUNCHER_PID}" >"${PID_PATH}"
echo "[DiffGui-Collect-50k-Resume] launcher_pid=${LAUNCHER_PID}"
echo "[DiffGui-Collect-50k-Resume] output=${OUTPUT_DIR}"
echo "[DiffGui-Collect-50k-Resume] wrapper_log=${WRAPPER_LOG}"
