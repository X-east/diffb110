#!/usr/bin/env bash
# Re-dock only the two fragment-affected models (DiffSBDD + TargetDiff) with the
# largest-fragment fix applied in dock_sdf_unidock.py (Path A). Other three
# models are single-fragment already, so their existing docking_results.csv is
# unchanged and stays comparable after summarize.
set -uo pipefail

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
EXP="${EXP:-${ROOT}/analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal}"
DOCK_SCRIPT="${DOCK_SCRIPT:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py}"
RECEPTOR="${RECEPTOR:-${ROOT}/data/8gjl.pdbqt}"
UNIDOCK_BIN="${UNIDOCK_BIN:-/opt/unidock-1.1.3/bin/unidock}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ENV_NAME="${ENV_NAME:-dif}"

mkdir -p "${EXP}/docking/logs"
MASTER_LOG="${EXP}/docking/logs/redock_fix_master.log"
: >"${MASTER_LOG}"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

# repeat|model|gpu|input_sdf
SPECS=(
  "rep1|DiffSBDD|0|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
  "rep1|TargetDiff|1|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
  "rep2|DiffSBDD|2|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
  "rep2|TargetDiff|3|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
  "rep3|DiffSBDD|4|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/diffsbdd.sanitize_valid.sdf"
  "rep3|TargetDiff|5|${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/targetdiff.sanitize_valid.sdf"
)

log "start redock_fix: 6 jobs (DiffSBDD+TargetDiff x 3 reps), 6 GPUs parallel"

for spec in "${SPECS[@]}"; do
  IFS='|' read -r repeat model gpu sdf <<<"${spec}"
  slug="$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/^_*//; s/_*$//')"
  out_dir="${EXP}/docking/${repeat}/${slug}"
  log_path="${EXP}/docking/logs/${repeat}_${slug}_redock_fix.log"
  log "launch ${repeat} ${model} gpu=${gpu} -> ${out_dir}"
  conda run -n "${ENV_NAME}" python "${DOCK_SCRIPT}" \
    --model "${model}" \
    --input-sdf "${sdf}" \
    --receptor "${RECEPTOR}" \
    --out-dir "${out_dir}" \
    --unidock-bin "${UNIDOCK_BIN}" \
    --gpu-id "${gpu}" \
    --batch-size "${BATCH_SIZE}" \
    >"${log_path}" 2>&1 &
done

wait
log "all redock_fix jobs finished"
