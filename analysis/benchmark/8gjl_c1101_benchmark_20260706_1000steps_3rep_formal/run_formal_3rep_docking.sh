#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
EXP="${EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal}"
DOCK_SCRIPT="${DOCK_SCRIPT:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${EXP}/summarize_formal_3rep_docking.py}"
ENV_NAME="${ENV_NAME:-dif}"
RECEPTOR="${RECEPTOR:-${ROOT}/data/8gjl.pdbqt}"
UNIDOCK_BIN="${UNIDOCK_BIN:-/opt/unidock-1.1.3/bin/unidock}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4}"
RUN_DOCKING="${RUN_DOCKING:-1}"

mkdir -p "${EXP}/docking/logs"
MASTER_LOG="${EXP}/docking/formal_3rep_docking_master.log"
: >"${MASTER_LOG}"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

gpu_for_index() {
  local index="$1"
  local -a gpus
  read -r -a gpus <<<"${GPU_IDS}"
  local n="${#gpus[@]}"
  echo "${gpus[$((index % n))]}"
}

run_one() {
  local index="$1"
  local repeat="$2"
  local model="$3"
  local sdf="$4"
  local gpu_id
  gpu_id="$(gpu_for_index "${index}")"
  local model_slug
  model_slug="$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/^_*//; s/_*$//')"
  local out_dir="${EXP}/docking/${repeat}/${model_slug}"
  local log_path="${EXP}/docking/logs/${repeat}_${model_slug}.log"
  local status_path="${EXP}/docking/logs/${repeat}_${model_slug}.status"

  if [[ -s "${out_dir}/docking_results.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "ok" >"${status_path}"
    log "[docking] skip existing ${repeat} ${model}: ${out_dir}/docking_results.csv"
    return 0
  fi

  log "[docking] run ${repeat} ${model} gpu=${gpu_id} sdf=${sdf}"
  if conda run -n "${ENV_NAME}" python "${DOCK_SCRIPT}" \
    --model "${model}" \
    --input-sdf "${sdf}" \
    --receptor "${RECEPTOR}" \
    --out-dir "${out_dir}" \
    --unidock-bin "${UNIDOCK_BIN}" \
    --gpu-id "${gpu_id}" \
    --batch-size "${BATCH_SIZE}" \
    >"${log_path}" 2>&1; then
    echo "ok" >"${status_path}"
    log "[docking] done ${repeat} ${model}"
  else
    echo "failed" >"${status_path}"
    log "[docking] FAILED ${repeat} ${model}; see ${log_path}"
    return 1
  fi
}

main() {
  log "[docking] started at $(date -Is)"
  log "[docking] exp=${EXP}"
  log "[docking] gpu_ids=${GPU_IDS} batch_size=${BATCH_SIZE}"

  local specs=(
    "rep1|DiffGui|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep1|DiffSBDD|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep1|PMDM|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep1|GCDM-SBDD|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep1|TargetDiff|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
    "rep2|DiffGui|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep2|DiffSBDD|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep2|PMDM|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep2|GCDM-SBDD|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep2|TargetDiff|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
    "rep3|DiffGui|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep3|DiffSBDD|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep3|PMDM|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep3|GCDM-SBDD|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep3|TargetDiff|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
  )

  if [[ "${RUN_DOCKING}" == "1" ]]; then
    local i=0
    for spec in "${specs[@]}"; do
      IFS="|" read -r repeat model sdf <<<"${spec}"
      run_one "${i}" "${repeat}" "${model}" "${sdf}"
      i=$((i + 1))
    done
  fi

  log "[docking] summarizing"
  conda run -n "${ENV_NAME}" python "${SUMMARY_SCRIPT}" --exp "${EXP}" | tee -a "${MASTER_LOG}"
  log "[docking] completed at $(date -Is)"
}

main "$@"
