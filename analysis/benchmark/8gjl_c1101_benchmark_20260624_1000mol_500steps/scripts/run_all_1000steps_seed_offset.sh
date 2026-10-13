#!/usr/bin/env bash
# Run a five-model 8GJL/C1101 benchmark replicate with seed offsets.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
REPLICATE="${REPLICATE:-2}"
SEED_OFFSET="${SEED_OFFSET:-7}"
TIMESTEPS="${TIMESTEPS:-1000}"
EXPECTED_TIMESTEPS="${EXPECTED_TIMESTEPS:-1000}"
EXP="${EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep${REPLICATE}_1000steps_seed$((SEED_OFFSET + 1))_$((SEED_OFFSET + 7))}"
INPUTS="${BASE_EXP}/inputs"
SCRIPTS="${BASE_EXP}/scripts"

TARGETDIFF_BATCH_SIZE="${TARGETDIFF_BATCH_SIZE:-5}"
RUN_DOCKING="${RUN_DOCKING:-0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
CHECK_OUTPUT_EMPTY="${CHECK_OUTPUT_EMPTY:-1}"
START_STAGE="${START_STAGE:-diffgui}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ATTEMPTS=1000

GPU_COUNT="${GPU_COUNT:-7}"
if [[ "${GPU_COUNT}" == "8" ]]; then
  # Logical shard IDs remain 1-based for the project seed convention; these
  # map to physical CUDA devices 0-7 in physical_gpu_for_shard().
  COUNTS=(125 125 125 125 125 125 125 125)
  GPUS=(1 2 3 4 5 6 7 8)
else
  COUNTS=(143 143 143 143 143 143 142)
  GPUS=(1 2 3 4 5 6 7)
fi
MODELS=(diffgui diffsbdd pmdm gcdm targetdiff)

MASTER_LOG="${EXP}/logs/all_rep${REPLICATE}_${TIMESTEPS}steps_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "[replicate] missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_no_existing_output() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    log "[replicate] ${label} already exists: ${path}"
    log "[replicate] refusing to overwrite existing outputs."
    exit 1
  fi
}

stage_rank() {
  case "$1" in
    diffgui) echo 1 ;;
    diffsbdd) echo 2 ;;
    pmdm) echo 3 ;;
    gcdm) echo 4 ;;
    targetdiff) echo 5 ;;
    evaluate) echo 6 ;;
    *) echo "[replicate] unknown START_STAGE=$1" >&2; exit 1 ;;
  esac
}

should_run_stage() {
  local stage="$1"
  [[ "$(stage_rank "${stage}")" -ge "$(stage_rank "${START_STAGE}")" ]]
}

seed_for_gpu() {
  local gpu_id="$1"
  echo $((gpu_id + SEED_OFFSET))
}

physical_gpu_for_shard() {
  local gpu_id="$1"
  if [[ "${GPU_COUNT}" == "8" ]]; then
    echo $((gpu_id - 1))
  else
    echo "${gpu_id}"
  fi
}

write_seed_plan() {
  local seed_plan="${EXP}/metadata/seed_plan.csv"
  mkdir -p "${EXP}/metadata"
  echo "replicate,model,gpu_id,count,seed,timesteps,output_dir,config_path,input_summary" >"${seed_plan}"
  for model in "${MODELS[@]}"; do
    for gpu_id in "${GPUS[@]}"; do
      local count="${COUNTS[$((gpu_id - 1))]}"
      local seed
      seed="$(seed_for_gpu "${gpu_id}")"
      local output_dir="${EXP}/outputs/${model}/gpu_${gpu_id}"
      local config_path=""
      local input_summary="8GJL/C1101"
      if [[ "${model}" == "diffgui" ]]; then
        config_path="${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml"
        input_summary="${INPUTS}/8GJL_C1101_pocket8A.pdb"
      elif [[ "${model}" == "targetdiff" ]]; then
        config_path="${EXP}/configs/targetdiff/targetdiff_c1101_gpu${gpu_id}.yml"
        input_summary="${INPUTS}/8GJL_C1101_protein_only.pdb + ${INPUTS}/8GJL_C1101_ligand.sdf"
      elif [[ "${model}" == "pmdm" ]]; then
        input_summary="${INPUTS}/8GJL_C1101_pocket8A.pdb"
      elif [[ "${model}" == "diffsbdd" ]]; then
        input_summary="${ROOT}/data/8GJL.pdb ref_ligand=C:1101"
      elif [[ "${model}" == "gcdm" ]]; then
        input_summary="${ROOT}/data/8GJL.pdb ref_ligand=C:1101"
      fi
      echo "${REPLICATE},${model},${gpu_id},${count},${seed},${TIMESTEPS},${output_dir},${config_path},${input_summary}" >>"${seed_plan}"
    done
  done
  cp "${seed_plan}" "${EXP}/run_manifest.csv"
}

write_diffgui_configs() {
  local cfg_dir="${EXP}/configs/diffgui"
  mkdir -p "${cfg_dir}"
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    local cfg="${cfg_dir}/diffgui_c1101_gpu${gpu_id}.yml"
    cat >"${cfg}" <<EOF
model:
  checkpoint: ckpt/trained.pt
  target: ${INPUTS}/8GJL_C1101_pocket8A.pdb
  ligand: None
  frag: None
  gen_mode: denovo
  logp: 2.0
  tpsa: 100
  sa: 1.0
  qed: 0.8
  aff: 12.0
bond_predictor: ckpt/bond_trained.pt
sample:
  seed: ${seed}
  batch_size: 2
  num_mols: ${count}
  num_timesteps: ${TIMESTEPS}
  save_traj_prob: 0.0
  sample: true
  sample_method: priori
  mode: pocket
  test_id: 0
  add_edge: None
  gui_strength: 3.0
  guidance:
  - uncertainty
  - 0.0001
data:
  name: protein_ligand
  path: data/PDBbind_v2020_pocket10
  split: data/PDBbind_pocket10_split.pt
  protein_root: data/PDBbind_v2020
  dataset: pdbbind
  transform:
    ligand_atom_mode: aromatic
    random_rot: false
    sample: false
EOF
  done
}

write_targetdiff_configs() {
  local cfg_dir="${EXP}/configs/targetdiff"
  mkdir -p "${cfg_dir}"
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    local result_path="${EXP}/outputs/targetdiff/gpu_${gpu_id}"
    local cfg="${cfg_dir}/targetdiff_c1101_gpu${gpu_id}.yml"
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
  done
}

preflight_static_check() {
  require_file "DiffGui sampler" "${ROOT}/reference project/代码/DiffGui/scripts/sample.py"
  require_file "DiffGui checkpoint" "${ROOT}/reference project/代码/DiffGui/ckpt/trained.pt"
  require_file "DiffGui bond checkpoint" "${ROOT}/reference project/代码/DiffGui/ckpt/bond_trained.pt"
  require_file "DiffSBDD checkpoint" "${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
  require_file "PMDM checkpoint" "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/PMDM/500.pt"
  require_file "GCDM checkpoint" "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/GCDM-SBDD/extracted/checkpoints/crossdocked_ca_joint_egnn.ckpt"
  require_file "TargetDiff checkpoint" "/home/md666/data/disk6/XDC/targetdiff/pretrained_models/pretrained_diffusion.pt"
  require_file "8GJL/C1101 pocket" "${INPUTS}/8GJL_C1101_pocket8A.pdb"
  require_file "8GJL/C1101 protein" "${INPUTS}/8GJL_C1101_protein_only.pdb"
  require_file "8GJL/C1101 ligand" "${INPUTS}/8GJL_C1101_ligand.sdf"
  require_file "8GJL PDB" "${ROOT}/data/8GJL.pdb"

  if [[ "${TIMESTEPS}" != "${EXPECTED_TIMESTEPS}" ]]; then
    log "[replicate] refusing to run with TIMESTEPS=${TIMESTEPS}; expected ${EXPECTED_TIMESTEPS}"
    exit 1
  fi

  local gpu_csv
  local count_csv
  gpu_csv="$(IFS=,; printf '%s' "${GPUS[*]}")"
  count_csv="$(IFS=,; printf '%s' "${COUNTS[*]}")"
  conda run -n dif python "${SCRIPTS}/validate_seed_plan.py" \
    --seed-plan "${EXP}/metadata/seed_plan.csv" \
    --gpus "${gpu_csv}" \
    --counts "${count_csv}" \
    --seed-offset "${SEED_OFFSET}" \
    --replicate "${REPLICATE}" \
    --timesteps "${TIMESTEPS}" \
    --previous-manifest "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7/run_manifest.csv" \
    --previous-manifest "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260626_pmdm_gcdm_1000steps/run_manifest.csv" \
    --previous-manifest "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/run_manifest.csv" \
    --previous-manifest "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/run_manifest.csv" \
    --previous-manifest "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/run_manifest.csv" \
    --summary-out "${EXP}/metadata/seed_plan.validation.json" \
    ${CHECK_OUTPUT_EMPTY:+$(if [[ "${CHECK_OUTPUT_EMPTY}" == "1" ]]; then echo "--check-output-empty"; fi)} | tee -a "${MASTER_LOG}"
}

run_parallel() {
  local model_name="$1"
  local log_dir="$2"
  shift 2
  mkdir -p "${log_dir}"

  local pids=()
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    local physical_gpu
    physical_gpu="$(physical_gpu_for_shard "${gpu_id}")"
    (
      export CUDA_VISIBLE_DEVICES="${physical_gpu}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}" "${seed}"
    ) >"${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    log "[${model_name}] launched shard=${gpu_id} physical_gpu=${physical_gpu} count=${count} seed=${seed} timesteps=${TIMESTEPS} pid=${pids[-1]}"
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

run_diffgui_one() {
  local gpu_id="$1"
  local _count="$2"
  local seed="$3"
  local project="${ROOT}/reference project/代码/DiffGui"
  local out_dir="${EXP}/outputs/diffgui/gpu_${gpu_id}"
  local cfg="${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml"
  local mpl_dir="${EXP}/mplconfig/diffgui/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  require_no_existing_output "DiffGui gpu_${gpu_id}" "${out_dir}/home_SDF"
  cd "${project}"
  export DIFFGUI_NUM_TIMESTEPS="${TIMESTEPS}"
  export PYTHONPATH="${BASE_EXP}/runtime_patch/diffgui:${project}${PYTHONPATH:+:${PYTHONPATH}}"
  export MPLCONFIGDIR="${mpl_dir}"
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  conda run -n diffgui python scripts/sample.py \
    --config "${cfg}" \
    --outdir "${out_dir}" \
    --device cuda:0 \
    --batch_size 2 \
    --fixed_attempts \
    --seed "${seed}"
}

aggregate_diffgui() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffGui \
    --output "${EXP}/aggregated/diffgui.sdf" \
    --limit "${ATTEMPTS}" \
    "${EXP}/outputs/diffgui" | tee -a "${MASTER_LOG}"
}

run_diffsbdd_one() {
  local gpu_id="$1"
  local count="$2"
  local seed="$3"
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
    --seed "${seed}" \
    --reconstruction-method openbabel
}

aggregate_diffsbdd() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffSBDD \
    --output "${EXP}/aggregated/diffsbdd.sdf" \
    --limit "${ATTEMPTS}" \
    "${EXP}/outputs/diffsbdd" | tee -a "${MASTER_LOG}"
}

run_pmdm_one() {
  local gpu_id="$1"
  local count="$2"
  local seed="$3"
  local out_dir="${EXP}/outputs/pmdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/pmdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  require_no_existing_output "PMDM gpu_${gpu_id}" "${out_dir}/pmdm_generated.sdf"
  export MPLCONFIGDIR="${mpl_dir}"
  conda run -n pmdm_modern_ref python \
    "${ROOT}/analysis/benchmark/modern_sbdd_20260531/run_pmdm_for_pdb.py" \
    --repo "${ROOT}/reference project/代码/modern_sbdd_20260531/PMDM" \
    --ckpt "${ROOT}/analysis/benchmark/modern_sbdd_20260531/assets/PMDM/500.pt" \
    --pdb-path "${INPUTS}/8GJL_C1101_pocket8A.pdb" \
    --outdir "${out_dir}" \
    --num-atom 24 \
    --num-samples "${count}" \
    --batch-size 32 \
    --n-steps "${TIMESTEPS}" \
    --device cuda \
    --seed "${seed}" \
    --fixed-attempts
}

aggregate_pmdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model PMDM \
    --output "${EXP}/aggregated/pmdm.sdf" \
    --limit "${ATTEMPTS}" \
    "${EXP}/outputs/pmdm" | tee -a "${MASTER_LOG}"
}

run_gcdm_one() {
  local gpu_id="$1"
  local count="$2"
  local seed="$3"
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
    --seed "${seed}"
}

aggregate_gcdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model GCDM-SBDD \
    --output "${EXP}/aggregated/gcdm.sdf" \
    --limit "${ATTEMPTS}" \
    "${EXP}/outputs/gcdm" | tee -a "${MASTER_LOG}"
}

run_targetdiff_one() {
  local gpu_id="$1"
  local _count="$2"
  local _seed="$3"
  local cfg="${EXP}/configs/targetdiff/targetdiff_c1101_gpu${gpu_id}.yml"
  cd "/home/md666/data/disk6/XDC/targetdiff"
  conda run -n targetdiff python scripts/sample_from_protein_ligand.py "${cfg}"
}

aggregate_targetdiff() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model TargetDiff \
    --output "${EXP}/aggregated/targetdiff.sdf" \
    --limit "${ATTEMPTS}" \
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

write_diffgui_attempt_summary() {
  conda run -n dif python "${SCRIPTS}/write_diffgui_attempt_summary.py" \
    --exp "${EXP}" \
    --attempts "${ATTEMPTS}" | tee -a "${MASTER_LOG}"
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
  mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated" "${EXP}/configs" "${EXP}/evaluation" "${EXP}/metadata" "${EXP}/mplconfig"
  : >"${MASTER_LOG}"

  log "[replicate] started at $(date -Is)"
  log "[replicate] exp=${EXP}"
  log "[replicate] base_exp=${BASE_EXP}"
  log "[replicate] replicate=${REPLICATE} seed_offset=${SEED_OFFSET} timesteps=${TIMESTEPS}"
  log "[replicate] precheck_only=${PRECHECK_ONLY} run_docking=${RUN_DOCKING} start_stage=${START_STAGE}"
  local physical_gpus
  physical_gpus="$(for gpu_id in "${GPUS[@]}"; do physical_gpu_for_shard "${gpu_id}"; done | tr '\n' ' ')"
  log "[replicate] logical_shards=${GPUS[*]} physical_gpus=${physical_gpus} counts=${COUNTS[*]} seed_rule=seed=gpu_id+${SEED_OFFSET}"

  write_diffgui_configs
  write_targetdiff_configs
  write_seed_plan
  preflight_static_check

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    log "[replicate] precheck completed; exiting before generation because PRECHECK_ONLY=1"
    return
  fi

  if should_run_stage "diffgui"; then
    log "[replicate] running DiffGui"
    run_parallel "diffgui" "${EXP}/logs/diffgui" run_diffgui_one
    aggregate_diffgui
  else
    require_file "existing DiffGui aggregate" "${EXP}/aggregated/diffgui.sdf"
    log "[replicate] skipping DiffGui; using existing aggregate ${EXP}/aggregated/diffgui.sdf"
  fi

  if should_run_stage "diffsbdd"; then
    log "[replicate] running DiffSBDD"
    run_parallel "diffsbdd" "${EXP}/logs/diffsbdd" run_diffsbdd_one
    aggregate_diffsbdd
  else
    require_file "existing DiffSBDD aggregate" "${EXP}/aggregated/diffsbdd.sdf"
    log "[replicate] skipping DiffSBDD; using existing aggregate ${EXP}/aggregated/diffsbdd.sdf"
  fi

  if should_run_stage "pmdm"; then
    log "[replicate] running PMDM"
    run_parallel "pmdm" "${EXP}/logs/pmdm" run_pmdm_one
    aggregate_pmdm
  else
    require_file "existing PMDM aggregate" "${EXP}/aggregated/pmdm.sdf"
    log "[replicate] skipping PMDM; using existing aggregate ${EXP}/aggregated/pmdm.sdf"
  fi

  if should_run_stage "gcdm"; then
    log "[replicate] running GCDM-SBDD"
    run_parallel "gcdm" "${EXP}/logs/gcdm" run_gcdm_one
    aggregate_gcdm
  else
    require_file "existing GCDM-SBDD aggregate" "${EXP}/aggregated/gcdm.sdf"
    log "[replicate] skipping GCDM-SBDD; using existing aggregate ${EXP}/aggregated/gcdm.sdf"
  fi

  if should_run_stage "targetdiff"; then
    log "[replicate] running TargetDiff"
    run_parallel "targetdiff" "${EXP}/logs/targetdiff" run_targetdiff_one
    aggregate_targetdiff
  else
    require_file "existing TargetDiff aggregate" "${EXP}/aggregated/targetdiff.sdf"
    log "[replicate] skipping TargetDiff; using existing aggregate ${EXP}/aggregated/targetdiff.sdf"
  fi

  log "[replicate] evaluating raw and sanitize-valid panels"
  evaluate_raw_and_valid
  write_diffgui_attempt_summary
  run_docking_if_requested

  log "[replicate] completed at $(date -Is)"
}

main "$@"
