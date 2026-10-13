#!/usr/bin/env bash
# Detached-friendly continuation launcher for five-model 8GJL/C1101 repeats.
set -euo pipefail

ROOT="${ROOT:-/data/disk6/DiffB110}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826_cont1}"
SCRIPT="${SCRIPT:-${RUN_ROOT}/script_snapshot/run_all_1000steps_seed_offset.sh}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826}"
STATUS_DIR="${RUN_ROOT}/status"
MASTER_LOG="${RUN_ROOT}/five_model_repeats_500steps_cont1.log"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

mkdir -p "${STATUS_DIR}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

wait_for_previous_completion() {
  local status_path="${PREVIOUS_ROOT}/status/rep6.status"
  log "waiting for completed rep6 status: ${status_path}"
  while true; do
    if [[ -s "${status_path}" ]] && grep -q '^status=completed$' "${status_path}"; then
      log "rep6 completion marker found"
      return 0
    fi
    if [[ -s "${status_path}" ]] && grep -q '^status=failed$' "${status_path}"; then
      log "rep6 failed; refusing to start continuation"
      return 1
    fi
    sleep "${WAIT_SECONDS}"
  done
}

run_one() {
  local replicate="$1"
  local seed_offset="$2"
  local exp="${RUN_ROOT}/rep${replicate}_seed$((seed_offset + 1))_$((seed_offset + 8))"
  local log_path="${RUN_ROOT}/rep${replicate}.log"
  local status_path="${STATUS_DIR}/rep${replicate}.status"
  local command_text
  command_text="GPU_COUNT=8 TIMESTEPS=500 EXPECTED_TIMESTEPS=500 REPLICATE=${replicate} SEED_OFFSET=${seed_offset} EXP=${exp} RUN_TAG=${RUN_TAG} PRECHECK_ONLY=0 GPU_COUNT=8 bash ${SCRIPT}"
  {
    echo "replicate=${replicate}"
    echo "seed_offset=${seed_offset}"
    echo "timesteps=500"
    echo "gpu_count=8"
    echo "attempts_per_model=1000"
    echo "logical_shards=1..8"
    echo "physical_gpus=0..7"
    echo "command=${command_text}"
    echo "started_at=$(date -Is)"
  } >"${status_path}"
  log "starting rep${replicate}: seeds=$((seed_offset + 1))-$((seed_offset + 8)) exp=${exp}"
  if env ROOT="${ROOT}" BASE_EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps" GPU_COUNT=8 TIMESTEPS=500 EXPECTED_TIMESTEPS=500 REPLICATE="${replicate}" SEED_OFFSET="${seed_offset}" EXP="${exp}" RUN_TAG="${RUN_TAG}" PRECHECK_ONLY=0 bash "${SCRIPT}" >"${log_path}" 2>&1; then
    echo "finished_at=$(date -Is)" >>"${status_path}"
    echo "status=completed" >>"${status_path}"
    log "completed rep${replicate}"
  else
    local exit_code="$?"
    echo "finished_at=$(date -Is)" >>"${status_path}"
    echo "status=failed" >>"${status_path}"
    echo "exit_code=${exit_code}" >>"${status_path}"
    log "FAILED rep${replicate} exit_code=${exit_code}; see ${log_path}"
    return "${exit_code}"
  fi
}

main() {
  : >"${MASTER_LOG}"
  log "five-model continuation started: T=500, 8 physical GPUs, 1000 attempts/model/repeat"
  log "models=DiffGui,DiffSBDD,PMDM,GCDM-SBDD,TargetDiff"
  log "repeats=rep7(seed46-53),rep8(seed54-61),rep9(seed62-69)"
  log "script=${SCRIPT}"
  log "previous_root=${PREVIOUS_ROOT}"
  bash -n "${SCRIPT}"
  log "bash syntax precheck passed"
  wait_for_previous_completion
  run_one 7 45
  run_one 8 53
  run_one 9 61
  log "five-model continuation completed"
}

main "$@"
