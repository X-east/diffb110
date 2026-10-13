#!/usr/bin/env bash
# Run only GCDM-SBDD and TargetDiff for the 8GJL/C1101 benchmark, then aggregate.
# This script does not rerun DiffGui, DiffSBDD, PMDM, docking, or evaluation.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"

TIMESTEPS="${TIMESTEPS:-500}"
TARGETDIFF_BATCH_SIZE="${TARGETDIFF_BATCH_SIZE:-5}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

COUNTS=(143 143 143 143 143 143 142)
GPUS=(1 2 3 4 5 6 7)

MASTER_LOG="${EXP}/logs/gcdm_targetdiff_full_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

require_no_existing_output() {
  local label="$1"
  local path="$2"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -s "${path}" ]]; then
    log "[full] ${label} output already exists: ${path}"
    log "[full] set ALLOW_OVERWRITE=1 only if you intentionally want to rerun/overwrite it."
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
    log "[${model_name}] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
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

run_gcdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/gcdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/gcdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  require_no_existing_output "GCDM-SBDD gpu_${gpu_id}" "${out_dir}/8GJL_mol.sdf"
  cd "${ROOT}/reference project/代码/modern_sbdd_20260531/GCDM-SBDD"
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

prepare_targetdiff_configs() {
  local cfg_dir="${EXP}/configs/targetdiff_full"
  mkdir -p "${cfg_dir}" "${EXP}/outputs/targetdiff"
  local manifest="${cfg_dir}/manifest.csv"
  echo "gpu_id,count,seed,config,result_path" >"${manifest}"
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    local seed="${gpu_id}"
    local result_path="${EXP}/outputs/targetdiff/gpu_${gpu_id}"
    local cfg="${cfg_dir}/targetdiff_c1101_gpu${gpu_id}.yml"
    mkdir -p "${result_path}"
    require_no_existing_output "TargetDiff gpu_${gpu_id}" "${result_path}/generated_molecules.sdf"
    cat >"${cfg}" <<EOF
model:
  checkpoint: /home/md666/data/disk6/XDC/targetdiff/pretrained_models/pretrained_diffusion.pt
input:
  protein_path: ${INPUTS}/8GJL_C1101_protein_only.pdb
  ligand_path: ${INPUTS}/8GJL_C1101_ligand.sdf
  pocket_radius: 10.0
runtime:
  device: cuda:0
  batch_size: ${TARGETDIFF_BATCH_SIZE}
  result_path: ${result_path}
sample:
  seed: ${seed}
  num_samples: ${count}
  num_steps: ${TIMESTEPS}
  pos_only: false
  center_pos_mode: protein
  sample_num_atoms: prior
EOF
    echo "${gpu_id},${count},${seed},${cfg},${result_path}" >>"${manifest}"
  done
  log "[targetdiff] wrote configs to ${cfg_dir}"
}

run_targetdiff_one() {
  local gpu_id="$1"
  local _count="$2"
  local cfg="${EXP}/configs/targetdiff_full/targetdiff_c1101_gpu${gpu_id}.yml"
  cd "/home/md666/data/disk6/XDC/targetdiff"
  conda run -n targetdiff python scripts/sample_from_protein_ligand.py "${cfg}"
}

aggregate_targetdiff() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model TargetDiff \
    --output "${EXP}/aggregated/targetdiff.sdf" \
    --limit 1000 \
    "${EXP}/outputs/targetdiff" | tee -a "${MASTER_LOG}"
}

main() {
  mkdir -p "${EXP}/logs" "${EXP}/outputs/gcdm" "${EXP}/outputs/targetdiff" "${EXP}/aggregated"
  : >"${MASTER_LOG}"
  log "[full] started at $(date -Is)"
  log "[full] timesteps=${TIMESTEPS} targetdiff_batch_size=${TARGETDIFF_BATCH_SIZE} allow_overwrite=${ALLOW_OVERWRITE}"
  log "[full] gpus=${GPUS[*]} counts=${COUNTS[*]}"

  log "[full] running GCDM-SBDD"
  run_parallel "gcdm" "${EXP}/logs/gcdm_full" run_gcdm_one
  log "[full] aggregating GCDM-SBDD"
  aggregate_gcdm

  log "[full] running TargetDiff"
  prepare_targetdiff_configs
  run_parallel "targetdiff" "${EXP}/logs/targetdiff_full" run_targetdiff_one
  log "[full] aggregating TargetDiff"
  aggregate_targetdiff

  log "[full] completed at $(date -Is)"
}

main "$@"
