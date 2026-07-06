#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
TARGETDIFF_ROOT="${TARGETDIFF_ROOT:-/home/md666/data/disk6/XDC/targetdiff}"
EXP="${EXP:-${ROOT}/analysis/benchmark/targetdiff_step_sweep_100mol_20260706}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/inputs}"

TARGETDIFF_ENV="${TARGETDIFF_ENV:-targetdiff}"
ANALYSIS_ENV="${ANALYSIS_ENV:-dif}"
GPU_ID="${GPU_ID:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-5}"
SEED="${SEED:-1}"
STEPS="${STEPS:-100 200 500 750 1000}"

mkdir -p "${EXP}/configs" "${EXP}/outputs" "${EXP}/logs" "${EXP}/summary"

MASTER_LOG="${EXP}/logs/step_sweep_master.log"
: >"${MASTER_LOG}"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

write_config() {
  local steps="$1"
  local result_path="${EXP}/outputs/steps_${steps}"
  local cfg="${EXP}/configs/targetdiff_c1101_100mol_steps_${steps}_seed${SEED}.yml"
  mkdir -p "${result_path}"
  cat >"${cfg}" <<EOF
model:
  checkpoint: ${TARGETDIFF_ROOT}/pretrained_models/pretrained_diffusion.pt
input:
  protein_path: ${INPUT_ROOT}/8GJL_C1101_protein_only.pdb
  ligand_path: ${INPUT_ROOT}/8GJL_C1101_ligand.sdf
  pocket_radius: 10.0
runtime:
  device: cuda:0
  batch_size: ${BATCH_SIZE}
  result_path: ${result_path}
sample:
  seed: ${SEED}
  num_samples: ${NUM_SAMPLES}
  num_steps: ${steps}
  pos_only: false
  center_pos_mode: protein
  sample_num_atoms: prior
EOF
  echo "${cfg}"
}

run_one() {
  local steps="$1"
  local cfg
  cfg="$(write_config "${steps}")"
  local log_path="${EXP}/logs/targetdiff_steps_${steps}.log"
  local status_path="${EXP}/logs/targetdiff_steps_${steps}.status"
  log "[step-sweep] running TargetDiff steps=${steps} samples=${NUM_SAMPLES} seed=${SEED} gpu=${GPU_ID}"
  if (
    cd "${TARGETDIFF_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${TARGETDIFF_ENV}" \
      python scripts/sample_from_protein_ligand.py "${cfg}"
  ) >"${log_path}" 2>&1; then
    echo "ok" >"${status_path}"
    log "[step-sweep] steps=${steps} completed"
  else
    echo "failed" >"${status_path}"
    log "[step-sweep] steps=${steps} FAILED; see ${log_path}"
  fi
}

main() {
  log "[step-sweep] started at $(date -Is)"
  log "[step-sweep] exp=${EXP}"
  log "[step-sweep] steps=${STEPS}"
  log "[step-sweep] num_samples=${NUM_SAMPLES} batch_size=${BATCH_SIZE} seed=${SEED} gpu=${GPU_ID}"
  for steps in ${STEPS}; do
    run_one "${steps}"
  done
  log "[step-sweep] summarizing"
  conda run -n "${ANALYSIS_ENV}" python "${EXP}/summarize_targetdiff_step_sweep.py" --exp "${EXP}" | tee -a "${MASTER_LOG}"
  log "[step-sweep] completed at $(date -Is)"
}

main "$@"
