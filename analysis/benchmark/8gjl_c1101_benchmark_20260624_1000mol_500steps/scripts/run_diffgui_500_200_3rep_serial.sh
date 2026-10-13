#!/usr/bin/env bash
# Run the planned DiffGui 500-step and 200-step 3-replicate matrix serially.
set -euo pipefail

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
SCRIPT="${BASE_EXP}/scripts/run_diffgui_step_replicate.sh"
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
START_FROM="${START_FROM:-1}"

mkdir -p "${SWEEP_ROOT}/logs"
MASTER_LOG="${SWEEP_ROOT}/logs/diffgui_500_200_3rep_serial_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

run_one() {
  local task_index="$1"
  shift
  local timesteps="$1"
  local replicate="$2"
  local seed_offset="$3"
  local exp="${SWEEP_ROOT}/${timesteps}steps_rep${replicate}_seed$((seed_offset + 1))_$((seed_offset + 7))"
  if [[ "${task_index}" -lt "${START_FROM}" ]]; then
    log "[serial] skipping task=${task_index} timesteps=${timesteps} replicate=${replicate} because START_FROM=${START_FROM}"
    return
  fi
  log "[serial] starting timesteps=${timesteps} replicate=${replicate} seed_offset=${seed_offset} exp=${exp}"
  TIMESTEPS="${timesteps}" \
    REPLICATE="${replicate}" \
    SEED_OFFSET="${seed_offset}" \
    EXP="${exp}" \
    PRECHECK_ONLY="${PRECHECK_ONLY}" \
    bash "${SCRIPT}" 2>&1 | tee -a "${MASTER_LOG}"
  log "[serial] finished timesteps=${timesteps} replicate=${replicate}"
}

main() {
  log "[serial] started at $(date -Is)"
  log "[serial] sweep_root=${SWEEP_ROOT}"
  log "[serial] precheck_only=${PRECHECK_ONLY}"
  log "[serial] start_from=${START_FROM}"
  log "[serial] matrix=500/200 steps x rep1-3, seeds 1-7/8-14/15-21, gpus 1-7"

  run_one 1 500 1 0
  run_one 2 500 2 7
  run_one 3 500 3 14
  run_one 4 200 1 0
  run_one 5 200 2 7
  run_one 6 200 3 14

  log "[serial] completed at $(date -Is)"
}

main "$@"
