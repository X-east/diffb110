#!/usr/bin/env bash
# Build a unified 8GJL/C1101 5-model benchmark with T=1000.
# Reuses existing 1000-step DiffGui/PMDM/GCDM-SBDD outputs and reruns only
# DiffSBDD plus TargetDiff with the original seed=gpu_id shard convention.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
PMDM_GCDM_EXP="${PMDM_GCDM_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260626_pmdm_gcdm_1000steps}"
EXP="${EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps}"
INPUTS="${BASE_EXP}/inputs"
SCRIPTS="${BASE_EXP}/scripts"

TIMESTEPS="${TIMESTEPS:-1000}"
TARGETDIFF_BATCH_SIZE="${TARGETDIFF_BATCH_SIZE:-5}"
RUN_DOCKING="${RUN_DOCKING:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

COUNTS=(143 143 143 143 143 143 142)
GPUS=(1 2 3 4 5 6 7)
MODELS=(diffgui diffsbdd pmdm gcdm targetdiff)

MASTER_LOG="${EXP}/logs/all_1000steps_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "[all1000] missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_no_existing_output() {
  local label="$1"
  local path="$2"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -s "${path}" ]]; then
    log "[all1000] ${label} already exists: ${path}"
    log "[all1000] set ALLOW_OVERWRITE=1 only if rerunning intentionally."
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
    log "[${model_name}] launched gpu=${gpu_id} count=${count} seed=${gpu_id} timesteps=${TIMESTEPS} pid=${pids[-1]}"
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

copy_reused_1000step_outputs() {
  log "[reuse] copying existing 1000-step DiffGui/PMDM/GCDM-SBDD aggregates"
  require_file "DiffGui aggregate" "${BASE_EXP}/aggregated/diffgui.sdf"
  require_file "PMDM 1000-step aggregate" "${PMDM_GCDM_EXP}/aggregated/pmdm.sdf"
  require_file "GCDM 1000-step aggregate" "${PMDM_GCDM_EXP}/aggregated/gcdm.sdf"

  cp "${BASE_EXP}/aggregated/diffgui.sdf" "${EXP}/aggregated/diffgui.sdf"
  cp "${PMDM_GCDM_EXP}/aggregated/pmdm.sdf" "${EXP}/aggregated/pmdm.sdf"
  cp "${PMDM_GCDM_EXP}/aggregated/gcdm.sdf" "${EXP}/aggregated/gcdm.sdf"

  {
    echo "model,source_sdf,target_sdf,diffusion_steps,seed_rule,notes"
    echo "DiffGui,${BASE_EXP}/aggregated/diffgui.sdf,${EXP}/aggregated/diffgui.sdf,1000,seed=gpu_id,reused existing DiffGui benchmark output"
    echo "PMDM,${PMDM_GCDM_EXP}/aggregated/pmdm.sdf,${EXP}/aggregated/pmdm.sdf,1000,seed=gpu_id,reused 20260626 PMDM 1000-step rerun"
    echo "GCDM-SBDD,${PMDM_GCDM_EXP}/aggregated/gcdm.sdf,${EXP}/aggregated/gcdm.sdf,1000,seed=gpu_id,reused 20260626 GCDM-SBDD 1000-step rerun"
  } >"${EXP}/metadata/reused_sources.csv"
}

write_run_manifest() {
  local manifest="${EXP}/run_manifest.csv"
  echo "model,gpu_id,count,seed,timesteps,output_dir" >"${manifest}"
  for model in diffsbdd targetdiff; do
    for gpu_id in "${GPUS[@]}"; do
      local count="${COUNTS[$((gpu_id - 1))]}"
      echo "${model},${gpu_id},${count},${gpu_id},${TIMESTEPS},${EXP}/outputs/${model}/gpu_${gpu_id}" >>"${manifest}"
    done
  done
}

run_diffsbdd_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffsbdd/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  require_no_existing_output "DiffSBDD gpu_${gpu_id}" "${out_dir}/generated.sdf"
  conda run -n dif python "${SCRIPTS}/run_diffsbdd_shard.py" \
    --repo-root "${ROOT}" \
    --checkpoint "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt" \
    --pdbfile "${ROOT}/data/8GJL.pdb" \
    --ref-ligand "C:1101" \
    --outfile "${out_dir}/generated.sdf" \
    --n-samples "${count}" \
    --batch-size 10 \
    --timesteps "${TIMESTEPS}" \
    --seed "${gpu_id}" \
    --reconstruction-method openbabel
}

aggregate_diffsbdd() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffSBDD \
    --output "${EXP}/aggregated/diffsbdd.sdf" \
    --limit 1000 \
    "${EXP}/outputs/diffsbdd" | tee -a "${MASTER_LOG}"
}

prepare_targetdiff_configs() {
  local cfg_dir="${EXP}/configs/targetdiff"
  mkdir -p "${cfg_dir}" "${EXP}/outputs/targetdiff"
  local manifest="${cfg_dir}/manifest.csv"
  echo "gpu_id,count,seed,timesteps,config,result_path" >"${manifest}"
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
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
  seed: ${gpu_id}
  num_samples: ${count}
  num_steps: ${TIMESTEPS}
  pos_only: false
  center_pos_mode: protein
  sample_num_atoms: prior
EOF
    echo "${gpu_id},${count},${gpu_id},${TIMESTEPS},${cfg},${result_path}" >>"${manifest}"
  done
  log "[targetdiff] wrote configs to ${cfg_dir}"
}

run_targetdiff_one() {
  local gpu_id="$1"
  local _count="$2"
  local cfg="${EXP}/configs/targetdiff/targetdiff_c1101_gpu${gpu_id}.yml"
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

evaluate_raw_and_valid() {
  local raw_eval_dir="${EXP}/evaluation/raw_all_models"
  local valid_dir="${EXP}/filtered_sanitize_valid"
  local valid_eval_dir="${EXP}/evaluation/sanitize_valid_only"

  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${raw_eval_dir}" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "PMDM=${EXP}/aggregated/pmdm.sdf" \
    --sdf "GCDM-SBDD=${EXP}/aggregated/gcdm.sdf" \
    --sdf "TargetDiff=${EXP}/aggregated/targetdiff.sdf" | tee -a "${MASTER_LOG}"

  conda run -n dif python "${SCRIPTS}/export_sanitize_valid_sdfs.py" \
    --out-dir "${valid_dir}" \
    --sdf "diffgui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "diffsbdd=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "pmdm=${EXP}/aggregated/pmdm.sdf" \
    --sdf "gcdm=${EXP}/aggregated/gcdm.sdf" \
    --sdf "targetdiff=${EXP}/aggregated/targetdiff.sdf" | tee -a "${MASTER_LOG}"

  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${valid_eval_dir}" \
    --sdf "diffgui=${valid_dir}/diffgui.sanitize_valid.sdf" \
    --sdf "diffsbdd=${valid_dir}/diffsbdd.sanitize_valid.sdf" \
    --sdf "pmdm=${valid_dir}/pmdm.sanitize_valid.sdf" \
    --sdf "gcdm=${valid_dir}/gcdm.sanitize_valid.sdf" \
    --sdf "targetdiff=${valid_dir}/targetdiff.sanitize_valid.sdf" | tee -a "${MASTER_LOG}"
}

run_docking_if_requested() {
  if [[ "${RUN_DOCKING}" != "1" ]]; then
    log "[docking] skipped; set RUN_DOCKING=1 to dock sanitize-valid SDFs."
    return
  fi

  local valid_dir="${EXP}/filtered_sanitize_valid"
  local dock_root="${EXP}/docking"
  mkdir -p "${dock_root}"
  for idx in "${!MODELS[@]}"; do
    local model="${MODELS[$idx]}"
    local gpu_id="${GPUS[$idx]}"
    conda run -n dif python "${SCRIPTS}/dock_sdf_unidock.py" \
      --model "${model}" \
      --input-sdf "${valid_dir}/${model}.sanitize_valid.sdf" \
      --receptor "${ROOT}/data/8gjl.pdbqt" \
      --out-dir "${dock_root}/${model}" \
      --gpu-id "${gpu_id}" \
      --batch-size 128 \
      --seed 42 | tee -a "${MASTER_LOG}"
  done

  conda run -n dif python "${SCRIPTS}/summarize_docking_panel.py" \
    --out-dir "${EXP}/evaluation/docking_summary" \
    --result "DiffGui=${dock_root}/diffgui/docking_results.csv" \
    --result "DiffSBDD=${dock_root}/diffsbdd/docking_results.csv" \
    --result "PMDM=${dock_root}/pmdm/docking_results.csv" \
    --result "GCDM-SBDD=${dock_root}/gcdm/docking_results.csv" \
    --result "TargetDiff=${dock_root}/targetdiff/docking_results.csv" | tee -a "${MASTER_LOG}"
}

main() {
  mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated" "${EXP}/configs" "${EXP}/evaluation" "${EXP}/metadata"
  : >"${MASTER_LOG}"

  log "[all1000] started at $(date -Is)"
  log "[all1000] exp=${EXP}"
  log "[all1000] timesteps=${TIMESTEPS} run_docking=${RUN_DOCKING} allow_overwrite=${ALLOW_OVERWRITE}"
  log "[all1000] gpus=${GPUS[*]} counts=${COUNTS[*]} seed_rule=seed=gpu_id"

  copy_reused_1000step_outputs
  write_run_manifest

  log "[all1000] running DiffSBDD T=${TIMESTEPS}"
  run_parallel "diffsbdd" "${EXP}/logs/diffsbdd" run_diffsbdd_one
  aggregate_diffsbdd

  log "[all1000] running TargetDiff T=${TIMESTEPS}"
  prepare_targetdiff_configs
  run_parallel "targetdiff" "${EXP}/logs/targetdiff" run_targetdiff_one
  aggregate_targetdiff

  log "[all1000] evaluating raw and sanitize-valid panels"
  evaluate_raw_and_valid
  run_docking_if_requested

  log "[all1000] completed at $(date -Is)"
}

main "$@"
