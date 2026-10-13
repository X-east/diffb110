#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-2 3 4 5 6 7}"
PIDS="${PIDS:-3107035 3107064 3106973 3107111 3107063 3106909}"
LOG_FILE="${LOG_FILE:-$ROOT/wait_and_resume_20260616.log}"

touch "$LOG_FILE"
echo "[wait-resume] start $(date '+%F %T')" >> "$LOG_FILE"
echo "[wait-resume] gpu_ids=${GPU_IDS}" >> "$LOG_FILE"
echo "[wait-resume] pids=${PIDS}" >> "$LOG_FILE"

for gpu_id in ${GPU_IDS}; do
    if [[ "$gpu_id" == "0" || "$gpu_id" == "1" ]]; then
        echo "[wait-resume] ERROR: GPU_IDS must not include GPU 0 or GPU 1" >> "$LOG_FILE"
        exit 1
    fi
done

while true; do
    active=0
    for pid in ${PIDS}; do
        if ps -p "$pid" -o args= 2>/dev/null | grep -q "scripts/sample.py" && \
           ps -p "$pid" -o args= 2>/dev/null | grep -q "outputs/01_zero"; then
            active=$((active + 1))
        fi
    done
    echo "[wait-resume] $(date '+%F %T') active_diffgui_children=${active}" >> "$LOG_FILE"
    if [[ "$active" == "0" ]]; then
        break
    fi
    sleep 120
done

echo "[wait-resume] aggregating 01_zero $(date '+%F %T')" >> "$LOG_FILE"
conda run -n dif python "$ROOT/aggregate_sdfs.py" \
    --group-dir "$ROOT/outputs/01_zero" \
    --gpu-ids ${GPU_IDS} \
    --out "$ROOT/outputs/01_zero/aggregated.sdf" \
    --manifest "$ROOT/outputs/01_zero/aggregated_manifest.csv" \
    >> "$LOG_FILE" 2>&1

echo "[wait-resume] resuming from 02_default $(date '+%F %T')" >> "$LOG_FILE"
GPU_IDS="${GPU_IDS}" START_AT=02_default bash "$ROOT/run_test.sh" \
    >> "$ROOT/run_test_20260616_from02.log" 2>&1
echo "[wait-resume] done $(date '+%F %T')" >> "$LOG_FILE"
