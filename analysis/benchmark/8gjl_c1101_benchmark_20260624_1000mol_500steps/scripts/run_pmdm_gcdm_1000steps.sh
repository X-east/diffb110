#!/usr/bin/env bash
# Rerun only PMDM and GCDM-SBDD for 8GJL/C1101 with T=1000.
# Keeps the original 500-step benchmark outputs untouched.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
EXP="${EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260626_pmdm_gcdm_1000steps}"
INPUTS="${BASE_EXP}/inputs"
SCRIPTS="${BASE_EXP}/scripts"

TIMESTEPS="${TIMESTEPS:-1000}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

COUNTS=(143 143 143 143 143 143 142)
GPUS=(1 2 3 4 5 6 7)

MASTER_LOG="${EXP}/logs/pmdm_gcdm_1000steps_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

require_no_existing_output() {
  local label="$1"
  local path="$2"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -s "${path}" ]]; then
    log "[rerun] ${label} output already exists: ${path}"
    log "[rerun] set ALLOW_OVERWRITE=1 only if rerunning intentionally."
    exit 1
  fi
}

run_parallel() {
  local model_name="$1"
  local log_dir="$2"
  shift 2
  mkdir -p "${log_dir}"

  local pids=()
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}"
    ) >"${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    log "[${model_name}] launched gpu=${gpu_id} count=${count} seed=${gpu_id} pid=${pids[-1]}"
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      log "[${model_name}] gpu=${GPUS[$idx]} completed"
    else
      log "[${model_name}] gpu=${GPUS[$idx]} FAILED (see ${log_dir}/gpu_${GPUS[$idx]}.log)"
      failed=1
    fi
  done
  return "${failed}"
}

write_manifest() {
  local manifest="${EXP}/run_manifest.csv"
  echo "model,gpu_id,count,seed,timesteps,output_dir" >"${manifest}"
  for model in pmdm gcdm; do
    for gpu_id in "${GPUS[@]}"; do
      local count="${COUNTS[$((gpu_id - 1))]}"
      echo "${model},${gpu_id},${count},${gpu_id},${TIMESTEPS},${EXP}/outputs/${model}/gpu_${gpu_id}" >>"${manifest}"
    done
  done
}

run_pmdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/pmdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/pmdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  require_no_existing_output "PMDM gpu_${gpu_id}" "${out_dir}/pmdm_generated.sdf"
  export MPLCONFIGDIR="${mpl_dir}"
  conda run -n pmdm_modern_ref python \
    "${ROOT}/analysis/benchmark/modern_sbdd_20260531/run_pmdm_for_pdb.py" \
    --repo "${ROOT}/reference project/modern_sbdd_20260531/PMDM" \
    --ckpt "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/PMDM/500.pt" \
    --pdb-path "${INPUTS}/8GJL_C1101_pocket8A.pdb" \
    --outdir "${out_dir}" \
    --num-atom 24 \
    --num-samples "${count}" \
    --batch-size 32 \
    --n-steps "${TIMESTEPS}" \
    --device cuda \
    --seed "${gpu_id}" \
    --fixed-attempts
}

aggregate_pmdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model PMDM \
    --output "${EXP}/aggregated/pmdm.sdf" \
    --limit 1000 \
    "${EXP}/outputs/pmdm" | tee -a "${MASTER_LOG}"
}

run_gcdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/gcdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/gcdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  require_no_existing_output "GCDM-SBDD gpu_${gpu_id}" "${out_dir}/8GJL_mol.sdf"
  cd "${ROOT}/reference project/modern_sbdd_20260531/GCDM-SBDD"
  export MPLCONFIGDIR="${mpl_dir}"
  conda run -n gcdm_sbdd_modern_ref python generate_ligands.py \
    "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/GCDM-SBDD/extracted/checkpoints/crossdocked_ca_joint_egnn.ckpt" \
    --pdbfile "${ROOT}/data/8GJL.pdb" \
    --ref_ligand "C:1101" \
    --outdir "${out_dir}" \
    --n_samples "${count}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${gpu_id}"
}

aggregate_gcdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model GCDM-SBDD \
    --output "${EXP}/aggregated/gcdm.sdf" \
    --limit 1000 \
    "${EXP}/outputs/gcdm" | tee -a "${MASTER_LOG}"
}

main() {
  mkdir -p "${EXP}/logs" "${EXP}/outputs/pmdm" "${EXP}/outputs/gcdm" "${EXP}/aggregated" "${EXP}/mplconfig"
  : >"${MASTER_LOG}"
  write_manifest

  log "[rerun] started at $(date -Is)"
  log "[rerun] base_exp=${BASE_EXP}"
  log "[rerun] exp=${EXP}"
  log "[rerun] timesteps=${TIMESTEPS} allow_overwrite=${ALLOW_OVERWRITE}"
  log "[rerun] gpus=${GPUS[*]} counts=${COUNTS[*]}"

  log "[rerun] running PMDM"
  run_parallel "pmdm" "${EXP}/logs/pmdm" run_pmdm_one
  log "[rerun] aggregating PMDM"
  aggregate_pmdm

  log "[rerun] running GCDM-SBDD"
  run_parallel "gcdm" "${EXP}/logs/gcdm" run_gcdm_one
  log "[rerun] aggregating GCDM-SBDD"
  aggregate_gcdm

  log "[rerun] completed at $(date -Is)"
}

main "$@"
