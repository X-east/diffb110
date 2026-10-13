#!/usr/bin/env bash
# Queue launcher: wait until the target GPUs are idle, then auto-launch the
# DiffGui continuation to epoch 100. This is a START-ONLY queue (it never
# interrupts a running training).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
UTIL_THRESH="${UTIL_THRESH:-20}"        # percent; consider idle if below this
FREE_MB_THRESH="${FREE_MB_THRESH:-6000}" # MB free memory required per GPU
POLL_SECONDS="${POLL_SECONDS:-120}"
NEED_STREAK="${NEED_STREAK:-3}"          # consecutive passes required
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$ROOT/launch/run_diffgui_deppa_full_lr1e5_wd1e4_b8x8_t200_100ep_continue_from85_20260911.sh}"
QUEUE_LOG="${QUEUE_LOG:-$ROOT/launch/archive/logs/queue_continue_from85_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "${QUEUE_LOG}")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"; }

log "queue start gpus=${GPUS} need util<${UTIL_THRESH}% and free>${FREE_MB_THRESH}MB x ${NEED_STREAK} consecutive polls (${POLL_SECONDS}s)"
log "will launch: ${LAUNCH_SCRIPT}"

streak=0
last_state=""
while true; do
  ok=1
  detail=""
  for g in ${GPUS//,/ }; do
    raw="$(nvidia-smi -i "$g" --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    util="$(printf '%s' "$raw" | awk -F, '{print $1}' | tr -cd '0-9')"
    free="$(printf '%s' "$raw" | awk -F, '{print $2}' | tr -cd '0-9')"
    if [ -z "${util:-}" ] || [ -z "${free:-}" ]; then ok=0; detail="gpu$g unreadable"; break; fi
    if [ "$util" -ge "$UTIL_THRESH" ] || [ "$free" -le "$FREE_MB_THRESH" ]; then
      ok=0; detail="gpu$g util=${util}% free=${free}MB"
    fi
  done

  if [ "$ok" -eq 1 ]; then
    streak=$((streak + 1))
    log "PASS ${streak}/${NEED_STREAK}"
    if [ "$streak" -ge "$NEED_STREAK" ]; then
      log "GPUs idle -> launching continuation in background"
      nohup bash "${LAUNCH_SCRIPT}" >>"${QUEUE_LOG}" 2>&1 &
      log "launched pid=$! ; exiting queue"
      exit 0
    fi
  else
    streak=0
    if [ "$detail" != "$last_state" ]; then log "waiting: ${detail}"; last_state="$detail"; fi
  fi
  sleep "${POLL_SECONDS}"
done
