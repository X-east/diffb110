#!/usr/bin/env bash
# Persistent monitor for the continuation launcher; it does not control jobs.
set -u

ROOT="${ROOT:-/data/disk6/DiffB110}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826_cont1}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826}"
LOG_PATH="${RUN_ROOT}/monitor_cont1.log"
STATUS_PATH="${RUN_ROOT}/monitor_cont1.status"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_PATH}"
}

replicate_exp() {
  local replicate="$1"
  case "${replicate}" in
    7) echo "${RUN_ROOT}/rep7_seed46_53" ;;
    8) echo "${RUN_ROOT}/rep8_seed54_61" ;;
    9) echo "${RUN_ROOT}/rep9_seed62_69" ;;
    *) return 1 ;;
  esac
}

launched_shards() {
  local replicate="$1"
  local log_path="${RUN_ROOT}/rep${replicate}.log"
  if [[ ! -s "${log_path}" ]]; then
    echo 0
    return
  fi
  grep -c '^\[diffgui\] launched shard=' "${log_path}" || true
}

active_diffgui_logs() {
  local replicate="$1"
  local exp
  exp="$(replicate_exp "${replicate}")"
  local count=0
  local gpu_id
  for gpu_id in 1 2 3 4 5 6 7 8; do
    if find "${exp}/outputs/diffgui/gpu_${gpu_id}" -type f -name log.txt -size +0c -print -quit 2>/dev/null | grep -q .; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

replicate_state() {
  local replicate="$1"
  local status_path="${RUN_ROOT}/status/rep${replicate}.status"
  local launch_count active_count
  launch_count="$(launched_shards "${replicate}")"
  active_count="$(active_diffgui_logs "${replicate}")"
  if [[ "${launch_count}" -ge 8 && "${active_count}" -ge 8 ]]; then
    echo "started_8_shards"
  elif [[ -s "${status_path}" ]] && grep -q '^status=failed$' "${status_path}"; then
    echo "failed"
  elif [[ -s "${status_path}" ]]; then
    echo "scheduled"
  else
    echo "waiting"
  fi
}

main() {
  : >"${LOG_PATH}"
  : >"${STATUS_PATH}"
  log "monitor started; continuation_pid=$(pgrep -f 'run_five_model_repeats_500steps_260826_cont1.sh' | head -1 || true)"
  while true; do
    local state7 state8 state9
    state7="$(replicate_state 7)"
    state8="$(replicate_state 8)"
    state9="$(replicate_state 9)"
    log "state rep7=${state7} launched=$(launched_shards 7)/8 active_logs=$(active_diffgui_logs 7)/8 rep8=${state8} launched=$(launched_shards 8)/8 active_logs=$(active_diffgui_logs 8)/8 rep9=${state9} launched=$(launched_shards 9)/8 active_logs=$(active_diffgui_logs 9)/8"

    if [[ "${state7}" == "failed" || "${state8}" == "failed" || "${state9}" == "failed" ]]; then
      echo "finished_at=$(date -Is)" >"${STATUS_PATH}"
      echo "status=failed" >>"${STATUS_PATH}"
      echo "rep7=${state7} rep8=${state8} rep9=${state9}" >>"${STATUS_PATH}"
      log "monitor stopped because a continuation replicate failed"
      return 1
    fi

    if [[ "${state7}" == "started_8_shards" && "${state8}" == "started_8_shards" && "${state9}" == "started_8_shards" ]]; then
      echo "finished_at=$(date -Is)" >"${STATUS_PATH}"
      echo "status=all_three_started" >>"${STATUS_PATH}"
      echo "rep7=8_shards rep8=8_shards rep9=8_shards" >>"${STATUS_PATH}"
      log "all three continuation replicates have started all 8 DiffGui shards"
      return 0
    fi
    sleep "${INTERVAL_SECONDS}"
  done
}

main "$@"
