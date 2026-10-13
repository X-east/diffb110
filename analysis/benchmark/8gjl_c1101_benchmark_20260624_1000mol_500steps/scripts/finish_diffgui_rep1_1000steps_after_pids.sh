#!/usr/bin/env bash
# Finish aggregation/evaluation after the detached DiffGui 1000-step samplers exit.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
EXP="${EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7}"
SCRIPTS="${BASE_EXP}/scripts"
ATTEMPTS="${ATTEMPTS:-1000}"
PIDS="${PIDS:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG="${EXP}/logs/finish_diffgui_rep1_1000steps_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${LOG}"
}

pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

wait_for_pids() {
  if [[ -z "${PIDS}" ]]; then
    log "[finish] no PIDS provided; skipping wait"
    return
  fi

  log "[finish] waiting for sampler pids: ${PIDS}"
  while true; do
    local running=()
    for pid in ${PIDS}; do
      if pid_running "${pid}"; then
        running+=("${pid}")
      fi
    done
    if [[ "${#running[@]}" -eq 0 ]]; then
      log "[finish] all sampler pids exited"
      return
    fi
    log "[finish] still running: ${running[*]}"
    sleep "${POLL_SECONDS}"
  done
}

check_sampler_outputs() {
  local found
  found="$(find "${EXP}/outputs/diffgui" -path '*/home_SDF/*.sdf' -type f 2>/dev/null | wc -l)"
  log "[finish] home_SDF sdf files found=${found}"
  if [[ "${found}" -lt 1 ]]; then
    log "[finish] no SDF files found; aborting aggregation"
    exit 1
  fi
}

aggregate_and_evaluate() {
  mkdir -p "${EXP}/aggregated" "${EXP}/evaluation/raw" "${EXP}/filtered_sanitize_valid" "${EXP}/evaluation/sanitize_valid_only"

  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffGui \
    --output "${EXP}/aggregated/diffgui.sdf" \
    --limit "${ATTEMPTS}" \
    "${EXP}/outputs/diffgui" | tee -a "${LOG}"

  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/raw" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" | tee -a "${LOG}"

  conda run -n dif python "${SCRIPTS}/export_sanitize_valid_sdfs.py" \
    --out-dir "${EXP}/filtered_sanitize_valid" \
    --sdf "diffgui=${EXP}/aggregated/diffgui.sdf" | tee -a "${LOG}"

  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/sanitize_valid_only" \
    --sdf "diffgui=${EXP}/filtered_sanitize_valid/diffgui.sanitize_valid.sdf" | tee -a "${LOG}"
}

write_attempt_summary() {
  conda run -n dif python "${SCRIPTS}/write_diffgui_attempt_summary.py" \
    --exp "${EXP}" \
    --attempts "${ATTEMPTS}" | tee -a "${LOG}"
}

main() {
  mkdir -p "${EXP}/logs"
  : >"${LOG}"
  log "[finish] started at $(date -Is)"
  log "[finish] exp=${EXP}"
  log "[finish] attempts=${ATTEMPTS}"
  wait_for_pids
  check_sampler_outputs
  aggregate_and_evaluate
  write_attempt_summary
  log "[finish] completed at $(date -Is)"
}

main "$@"
