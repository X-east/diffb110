#!/usr/bin/env bash
# Run one DiffGui 8GJL/C1101 replicate with an explicit sampling step count.
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
BASE_EXP="${BASE_EXP:-${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps}"
TIMESTEPS="${TIMESTEPS:-500}"
REPLICATE="${REPLICATE:-1}"
SEED_OFFSET="${SEED_OFFSET:-$(((REPLICATE - 1) * 7))}"
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707}"
EXP="${EXP:-${SWEEP_ROOT}/${TIMESTEPS}steps_rep${REPLICATE}_seed$((SEED_OFFSET + 1))_$((SEED_OFFSET + 7))}"
INPUTS="${BASE_EXP}/inputs"
SCRIPTS="${BASE_EXP}/scripts"

PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
CHECK_OUTPUT_EMPTY="${CHECK_OUTPUT_EMPTY:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ATTEMPTS=1000

COUNTS=(143 143 143 143 143 143 142)
GPUS=(1 2 3 4 5 6 7)

MASTER_LOG="${EXP}/logs/diffgui_${TIMESTEPS}steps_rep${REPLICATE}_${RUN_TAG}.log"

log() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "[diffgui-step] missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_no_existing_output() {
  local label="$1"
  local path="$2"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${path}" ]]; then
    log "[diffgui-step] ${label} already exists: ${path}"
    log "[diffgui-step] set ALLOW_OVERWRITE=1 only if rerunning intentionally."
    exit 1
  fi
}

seed_for_gpu() {
  local gpu_id="$1"
  echo $((gpu_id + SEED_OFFSET))
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

write_manifest() {
  local manifest="${EXP}/run_manifest.csv"
  local seed_plan="${EXP}/metadata/seed_plan.csv"
  mkdir -p "${EXP}/metadata"
  echo "replicate,model,gpu_id,count,seed,timesteps,output_dir,config_path,input_summary" >"${seed_plan}"
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    echo "${REPLICATE},diffgui,${gpu_id},${count},${seed},${TIMESTEPS},${EXP}/outputs/diffgui/gpu_${gpu_id},${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml,${INPUTS}/8GJL_C1101_pocket8A.pdb" >>"${seed_plan}"
  done
  cp "${seed_plan}" "${manifest}"
}

preflight_static_check() {
  require_file "DiffGui sampler" "${ROOT}/reference project/代码/DiffGui/scripts/sample.py"
  require_file "DiffGui checkpoint" "${ROOT}/reference project/代码/DiffGui/ckpt/trained.pt"
  require_file "DiffGui bond checkpoint" "${ROOT}/reference project/代码/DiffGui/ckpt/bond_trained.pt"
  require_file "8GJL/C1101 pocket" "${INPUTS}/8GJL_C1101_pocket8A.pdb"

  if [[ "${TIMESTEPS}" -le 0 ]]; then
    log "[diffgui-step] TIMESTEPS must be positive, got ${TIMESTEPS}"
    exit 1
  fi
  if [[ "${TIMESTEPS}" != "500" && "${TIMESTEPS}" != "200" && "${TIMESTEPS}" != "100" ]]; then
    log "[diffgui-step] refusing TIMESTEPS=${TIMESTEPS}; this sweep is restricted to 500, 200, or 100."
    exit 1
  fi

  conda run -n dif python "${SCRIPTS}/validate_seed_plan.py" \
    --seed-plan "${EXP}/metadata/seed_plan.csv" \
    --models diffgui \
    --seed-offset "${SEED_OFFSET}" \
    --replicate "${REPLICATE}" \
    --timesteps "${TIMESTEPS}" \
    --summary-out "${EXP}/metadata/seed_plan.validation.json" \
    ${CHECK_OUTPUT_EMPTY:+$(if [[ "${CHECK_OUTPUT_EMPTY}" == "1" ]]; then echo "--check-output-empty"; fi)} | tee -a "${MASTER_LOG}"
}

run_parallel() {
  local log_dir="${EXP}/logs/diffgui"
  mkdir -p "${log_dir}"

  local pids=()
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[$((gpu_id - 1))]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      run_diffgui_one "${gpu_id}" "${count}" "${seed}"
    ) >"${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    log "[diffgui] launched gpu=${gpu_id} count=${count} seed=${seed} timesteps=${TIMESTEPS} pid=${pids[-1]}"
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      log "[diffgui] gpu=${GPUS[$idx]} completed"
    else
      log "[diffgui] gpu=${GPUS[$idx]} FAILED (see ${log_dir}/gpu_${GPUS[$idx]}.log)"
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
  export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
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

evaluate_diffgui() {
  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/raw" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" | tee -a "${MASTER_LOG}"

  conda run -n dif python "${SCRIPTS}/export_sanitize_valid_sdfs.py" \
    --out-dir "${EXP}/filtered_sanitize_valid" \
    --sdf "diffgui=${EXP}/aggregated/diffgui.sdf" | tee -a "${MASTER_LOG}"

  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/sanitize_valid_only" \
    --sdf "diffgui=${EXP}/filtered_sanitize_valid/diffgui.sanitize_valid.sdf" | tee -a "${MASTER_LOG}"
}

write_attempt_summary() {
  conda run -n dif python "${SCRIPTS}/write_diffgui_attempt_summary.py" \
    --exp "${EXP}" \
    --attempts "${ATTEMPTS}" \
    --diffusion-steps "${TIMESTEPS}" \
    --source-note "DiffGui ${TIMESTEPS}-step replicate with explicit sample.num_timesteps and DIFFGUI_NUM_TIMESTEPS" | tee -a "${MASTER_LOG}"
}

verify_timestep_logs() {
  local missing=0
  for gpu_id in "${GPUS[@]}"; do
    local log_file="${EXP}/logs/diffgui/gpu_${gpu_id}.log"
    local output_log
    output_log="$(find "${EXP}/outputs/diffgui/gpu_${gpu_id}" -maxdepth 2 -name log.txt -print -quit 2>/dev/null || true)"
    if grep -q "Using DiffGui sampling timesteps override: ${TIMESTEPS} from sample.num_timesteps" "${log_file}"; then
      continue
    fi
    if [[ -n "${output_log}" ]] && grep -q "Using DiffGui sampling timesteps override: ${TIMESTEPS} from sample.num_timesteps" "${output_log}"; then
      continue
    fi
    if [[ -n "${output_log}" ]]; then
      log "[verify] missing timestep override evidence in ${log_file} and ${output_log}"
    else
      log "[verify] missing timestep override evidence in ${log_file}; no DiffGui output log found"
    fi
      missing=1
  done
  if [[ "${missing}" != "0" ]]; then
    exit 1
  fi
  log "[verify] all DiffGui GPU logs confirm sample.num_timesteps=${TIMESTEPS}"
}

main() {
  mkdir -p "${EXP}/logs" "${EXP}/outputs/diffgui" "${EXP}/aggregated" "${EXP}/configs" "${EXP}/evaluation" "${EXP}/filtered_sanitize_valid" "${EXP}/metadata" "${EXP}/mplconfig"
  : >"${MASTER_LOG}"

  log "[diffgui-step] started at $(date -Is)"
  log "[diffgui-step] exp=${EXP}"
  log "[diffgui-step] sweep_root=${SWEEP_ROOT}"
  log "[diffgui-step] base_exp=${BASE_EXP}"
  log "[diffgui-step] replicate=${REPLICATE} seed_offset=${SEED_OFFSET} timesteps=${TIMESTEPS}"
  log "[diffgui-step] precheck_only=${PRECHECK_ONLY} allow_overwrite=${ALLOW_OVERWRITE}"
  log "[diffgui-step] gpus=${GPUS[*]} counts=${COUNTS[*]} seed_rule=seed=gpu_id+${SEED_OFFSET}"

  write_diffgui_configs
  write_manifest
  preflight_static_check

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    log "[diffgui-step] precheck completed; exiting before generation because PRECHECK_ONLY=1"
    return
  fi

  run_parallel
  verify_timestep_logs
  aggregate_diffgui
  evaluate_diffgui
  write_attempt_summary

  log "[diffgui-step] completed at $(date -Is)"
}

main "$@"
