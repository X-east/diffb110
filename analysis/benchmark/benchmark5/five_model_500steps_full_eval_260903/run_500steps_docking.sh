#!/usr/bin/env bash
# Run UniDock docking for the 500-step five-model benchmark (rep4-rep8).
#
# Parameters are intentionally identical to the 1000-step formal run so the two
# step budgets are directly comparable:
#   receptor   data/8gjl.pdbqt
#   box        center (207.293, 212.43, 172.771), size 20 A
#   scoring    vina      search_mode  fast      batch_size 128
set -euo pipefail

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
EXP="${EXP:-${ROOT}/analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903}"
DOCK_SCRIPT="${DOCK_SCRIPT:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py}"
ENV_NAME="${ENV_NAME:-dif}"
RECEPTOR="${RECEPTOR:-${ROOT}/data/8gjl.pdbqt}"
UNIDOCK_BIN="${UNIDOCK_BIN:-/opt/unidock-1.1.3/bin/unidock}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_DOCKING="${RUN_DOCKING:-1}"

R1="${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
R2="${ROOT}/analysis/benchmark/benchmark5/five_model_repeats_500steps_260826_cont1"

mkdir -p "${EXP}/docking/logs"
MASTER_LOG="${EXP}/docking/docking_500steps_master.log"
: >"${MASTER_LOG}"

log() { echo "$@" | tee -a "${MASTER_LOG}"; }

gpu_for_index() {
  local index="$1"; local -a gpus
  read -r -a gpus <<<"${GPU_IDS}"
  local n="${#gpus[@]}"
  echo "${gpus[$((index % n))]}"
}

run_one() {
  local index="$1" repeat="$2" model="$3" sdf="$4"
  local gpu_id; gpu_id="$(gpu_for_index "${index}")"
  local model_slug
  model_slug="$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/^_*//; s/_*$//')"
  local out_dir="${EXP}/docking/${repeat}/${model_slug}"
  local log_path="${EXP}/docking/logs/${repeat}_${model_slug}.log"
  local status_path="${EXP}/docking/logs/${repeat}_${model_slug}.status"

  if [[ -s "${out_dir}/docking_results.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "ok" >"${status_path}"
    log "[docking] skip existing ${repeat} ${model}"
    return 0
  fi

  log "[docking] run ${repeat} ${model} gpu=${gpu_id}"
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
    "rep4|DiffGui|${R1}/rep4_seed22_29/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep4|DiffSBDD|${R1}/rep4_seed22_29/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep4|PMDM|${R1}/rep4_seed22_29/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep4|GCDM-SBDD|${R1}/rep4_seed22_29/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep4|TargetDiff|${R1}/rep4_seed22_29/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
    "rep5|DiffGui|${R1}/rep5_seed30_37/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep5|DiffSBDD|${R1}/rep5_seed30_37/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep5|PMDM|${R1}/rep5_seed30_37/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep5|GCDM-SBDD|${R1}/rep5_seed30_37/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep5|TargetDiff|${R1}/rep5_seed30_37/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
    "rep6|DiffGui|${R1}/rep6_seed38_45/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep6|DiffSBDD|${R1}/rep6_seed38_45/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep6|PMDM|${R1}/rep6_seed38_45/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep6|GCDM-SBDD|${R1}/rep6_seed38_45/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep6|TargetDiff|${R1}/rep6_seed38_45/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
    "rep7|DiffGui|${R2}/rep7_seed46_53/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep7|DiffSBDD|${R2}/rep7_seed46_53/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep7|PMDM|${R2}/rep7_seed46_53/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep7|GCDM-SBDD|${R2}/rep7_seed46_53/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep7|TargetDiff|${R2}/rep7_seed46_53/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
    "rep8|DiffGui|${R2}/rep8_seed54_61/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
    "rep8|DiffSBDD|${R2}/rep8_seed54_61/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
    "rep8|PMDM|${R2}/rep8_seed54_61/filtered_sanitize_valid/pmdm.sanitize_valid.sdf"
    "rep8|GCDM-SBDD|${R2}/rep8_seed54_61/filtered_sanitize_valid/gcdm.sanitize_valid.sdf"
    "rep8|TargetDiff|${R2}/rep8_seed54_61/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
  )

  if [[ "${RUN_DOCKING}" == "1" ]]; then
    local i=0
    for spec in "${specs[@]}"; do
      IFS="|" read -r repeat model sdf <<<"${spec}"
      run_one "${i}" "${repeat}" "${model}" "${sdf}"
      i=$((i + 1))
    done
  fi

  log "[docking] completed at $(date -Is)"
}

main "$@"
