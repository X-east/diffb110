#!/usr/bin/env bash
# Detached launcher for three serial five-model benchmark repeats at T=1000.
set -euo pipefail

ROOT="${ROOT:-/data/disk6/DiffB110}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_1000steps_260827}"
SCRIPT="${SCRIPT:-${RUN_ROOT}/script_snapshot/run_all_1000steps_seed_offset.sh}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
STATUS_DIR="${RUN_ROOT}/status"
MASTER_LOG="${RUN_ROOT}/three_repeats_1000steps.log"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${STATUS_DIR}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

run_one() {
  local replicate="$1"
  local seed_offset="$2"
  local exp="${RUN_ROOT}/rep${replicate}_seed$((seed_offset + 1))_$((seed_offset + 8))"
  local log_path="${RUN_ROOT}/rep${replicate}.log"
  local status_path="${STATUS_DIR}/rep${replicate}.status"
  local command_text
  command_text="ROOT=${ROOT} BASE_EXP=${BASE_EXP} GPU_COUNT=8 TIMESTEPS=1000 EXPECTED_TIMESTEPS=1000 REPLICATE=${replicate} SEED_OFFSET=${seed_offset} EXP=${exp} RUN_TAG=${RUN_TAG} RUN_DOCKING=0 PRECHECK_ONLY=0 bash ${SCRIPT}"
  {
    echo "replicate=${replicate}"
    echo "seed_offset=${seed_offset}"
    echo "timesteps=1000"
    echo "gpu_count=8"
    echo "attempts_per_model=1000"
    echo "logical_shards=1..8"
    echo "physical_gpus=0..7"
    echo "command=${command_text}"
    echo "started_at=$(date -Is)"
  } >"${status_path}"
  log "starting rep${replicate}: seeds=$((seed_offset + 1))-$((seed_offset + 8)) exp=${exp}"
  if env ROOT="${ROOT}" BASE_EXP="${BASE_EXP}" GPU_COUNT=8 TIMESTEPS=1000 EXPECTED_TIMESTEPS=1000 REPLICATE="${replicate}" SEED_OFFSET="${seed_offset}" EXP="${exp}" RUN_TAG="${RUN_TAG}" RUN_DOCKING=0 PRECHECK_ONLY=0 bash "${SCRIPT}" >"${log_path}" 2>&1; then
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
  log "three five-model repeats started: T=1000, 8 physical GPUs, 1000 attempts/model/repeat"
  log "models=DiffGui,DiffSBDD,PMDM,GCDM-SBDD,TargetDiff"
  log "repeats=rep10(seed70-77),rep11(seed78-85),rep12(seed86-93)"
  log "script=${SCRIPT}"
  bash -n "${SCRIPT}"
  log "bash syntax precheck passed"
  run_one 10 69
  run_one 11 77
  run_one 12 85
  log "three five-model repeats completed"
}

main "$@"
