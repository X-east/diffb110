#!/usr/bin/env bash
# Independent generation with the project-trained DiffGui DePPA-full epoch80 checkpoint.
#
# Settings: 500 diffusion steps, 6 GPUs, 2000 total fixed generation attempts.
# Fixed attempts means each shard stops after its configured attempt quota and
# invalid or failed molecules are never refilled.
#
# Usage:
#   PRECHECK_ONLY=1 bash launch/run_diffgui_epoch80_500steps_6gpu_2000attempts_20261009.sh
#   bash launch/run_diffgui_epoch80_500steps_6gpu_2000attempts_20261009.sh
set -euo pipefail

source /home/md666/miniforge3/etc/profile.d/conda.sh

ROOT="${ROOT:-/home/md666/data/disk6/DiffB110}"
RUN_TAG="${RUN_TAG:-20261009_214739}"
EXP="${EXP:-${ROOT}/OUT/diffgui_epoch80_independent_500steps_6gpu_2000attempts_${RUN_TAG}}"
PROJECT="${ROOT}/reference project/代码/DiffGui"
BASE_SCRIPTS="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts"
INPUTS="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/inputs"

TIMESTEPS="${TIMESTEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GUI_STRENGTH="${GUI_STRENGTH:-3.0}"
SEED_OFFSET="${SEED_OFFSET:-42}"
TOTAL_ATTEMPTS="${TOTAL_ATTEMPTS:-2000}"
# Follow-up batches reuse the already-exported snapshot instead of re-exporting it.
SNAPSHOT_CHECKPOINT="${SNAPSHOT_CHECKPOINT:-${ROOT}/OUT/diffgui_epoch80_independent_500steps_6gpu_2000attempts_20261009_214739/checkpoints/diffgui_epoch80_reference_format.pt}"
EXPORT_REPORT="${EXPORT_REPORT:-${ROOT}/OUT/diffgui_epoch80_independent_500steps_6gpu_2000attempts_20261009_214739/metadata/checkpoint_export_report.json}"
CHECKPOINT="${CHECKPOINT:-${SNAPSHOT_CHECKPOINT}}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
CHECK_OUTPUT_EMPTY="${CHECK_OUTPUT_EMPTY:-1}"
START_SEC="$(date +%s)"

# Logical shard ids follow the project 1-based seed convention.
# These map to physical CUDA devices 0-5 through physical_gpu_for_shard().
# GPU_LIST / COUNT_LIST exist so a short single-shard smoke can reuse this script.
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6}"
COUNT_LIST="${COUNT_LIST:-334 334 333 333 333 333}"
read -r -a GPUS <<<"${GPU_LIST}"
read -r -a COUNTS <<<"${COUNT_LIST}"

MASTER_LOG="${EXP}/logs/generation.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    log "missing ${label}: ${path}"
    exit 1
  fi
}

physical_gpu_for_shard() {
  local gpu_id="$1"
  echo $((gpu_id - 1))
}

seed_for_gpu() {
  local gpu_id="$1"
  echo $((gpu_id + SEED_OFFSET))
}

write_configs() {
  local cfg_dir="${EXP}/configs/diffgui"
  mkdir -p "${cfg_dir}"
  for idx in "${!GPUS[@]}"; do
    local gpu_id="${GPUS[$idx]}"
    local count="${COUNTS[$idx]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    local cfg="${cfg_dir}/diffgui_c1101_gpu${gpu_id}.yml"
    cat >"${cfg}" <<EOF
model:
  checkpoint: ${CHECKPOINT}
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
  batch_size: ${BATCH_SIZE}
  num_mols: ${count}
  num_timesteps: ${TIMESTEPS}
  save_traj_prob: 0.0
  sample: true
  sample_method: priori
  mode: pocket
  test_id: 0
  add_edge: None
  gui_strength: ${GUI_STRENGTH}
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

write_seed_plan() {
  local seed_plan="${EXP}/metadata/seed_plan.csv"
  mkdir -p "${EXP}/metadata"
  echo "model,gpu_id,count,seed,timesteps,output_dir,config_path,input_summary,checkpoint" >"${seed_plan}"
  for idx in "${!GPUS[@]}"; do
    local gpu_id="${GPUS[$idx]}"
    local count="${COUNTS[$idx]}"
    local seed
    seed="$(seed_for_gpu "${gpu_id}")"
    echo "diffgui,${gpu_id},${count},${seed},${TIMESTEPS},${EXP}/outputs/diffgui/gpu_${gpu_id},${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml,${INPUTS}/8GJL_C1101_pocket8A.pdb,${CHECKPOINT}" >>"${seed_plan}"
  done
  cp "${seed_plan}" "${EXP}/run_manifest.csv"
}

previous_manifests() {
  local candidates=(
    "${ROOT}/OUT/diffgui_epoch80_independent_500steps_6gpu_2000attempts_20261009_214739/run_manifest.csv"
    "${ROOT}/analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707/500steps_rep1_seed1_7/run_manifest.csv"
    "${ROOT}/analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/500steps_rep2_seed8_14/run_manifest.csv"
    "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7/run_manifest.csv"
    "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/run_manifest.csv"
    "${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/run_manifest.csv"
  )
  local args=()
  for path in "${candidates[@]}"; do
    [[ -f "${path}" ]] && args+=("--previous-manifest" "${path}")
  done
  printf '%s\n' ${args[@]+"${args[@]}"}
}

preflight() {
  require_file "DiffGui sampler" "${PROJECT}/scripts/sample.py"
  require_file "DiffGui bond checkpoint" "${PROJECT}/ckpt/bond_trained.pt"
  require_file "epoch80 snapshot checkpoint" "${CHECKPOINT}"
  require_file "checkpoint export report" "${EXPORT_REPORT}"
  require_file "8GJL/C1101 pocket" "${INPUTS}/8GJL_C1101_pocket8A.pdb"
  require_file "seed plan" "${EXP}/metadata/seed_plan.csv"

  if [[ "${TIMESTEPS}" != "500" ]]; then
    log "refusing TIMESTEPS=${TIMESTEPS}; this run plan is fixed to 500 steps"
    exit 1
  fi
  local count_sum=0
  for count in "${COUNTS[@]}"; do
    count_sum=$((count_sum + count))
  done
  if [[ "${count_sum}" != "${TOTAL_ATTEMPTS}" ]]; then
    log "shard counts sum to ${count_sum}, expected ${TOTAL_ATTEMPTS}"
    exit 1
  fi
  if [[ "${#GPUS[@]}" != "${#COUNTS[@]}" ]]; then
    log "GPUS and COUNTS length mismatch: ${#GPUS[@]} shards vs ${#COUNTS[@]} counts"
    exit 1
  fi

  local gpu_csv count_csv
  gpu_csv="$(IFS=,; printf '%s' "${GPUS[*]}")"
  count_csv="$(IFS=,; printf '%s' "${COUNTS[*]}")"
  local previous=()
  mapfile -t previous < <(previous_manifests)
  conda run -n dif python "${BASE_SCRIPTS}/validate_seed_plan.py" \
    --seed-plan "${EXP}/metadata/seed_plan.csv" \
    --models diffgui \
    --gpus "${gpu_csv}" \
    --counts "${count_csv}" \
    --seed-offset "${SEED_OFFSET}" \
    --timesteps "${TIMESTEPS}" \
    --summary-out "${EXP}/metadata/seed_plan.validation.json" \
    ${previous[@]+"${previous[@]}"} \
    ${CHECK_OUTPUT_EMPTY:+$(if [[ "${CHECK_OUTPUT_EMPTY}" == "1" ]]; then echo "--check-output-empty"; fi)} \
    | tee -a "${MASTER_LOG}" || {
      log "seed plan validation failed"
      exit 1
    }
}

run_diffgui_one() {
  local gpu_id="$1"
  local _count="$2"
  local seed="$3"
  local out_dir="${EXP}/outputs/diffgui/gpu_${gpu_id}"
  local cfg="${EXP}/configs/diffgui/diffgui_c1101_gpu${gpu_id}.yml"
  local mpl_dir="${EXP}/mplconfig/diffgui/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  cd "${PROJECT}"
  unset DIFFGUI_NUM_TIMESTEPS
  export CUDA_VISIBLE_DEVICES="$(physical_gpu_for_shard "${gpu_id}")"
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
  export MPLCONFIGDIR="${mpl_dir}"
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  conda run -n diffgui python scripts/sample.py \
    --config "${cfg}" \
    --outdir "${out_dir}" \
    --device cuda:0 \
    --batch_size "${BATCH_SIZE}" \
    --fixed_attempts \
    --seed "${seed}"
}

run_parallel() {
  local log_dir="${EXP}/logs/diffgui"
  mkdir -p "${log_dir}"
  local pids=()
  for idx in "${!GPUS[@]}"; do
    local gpu_id="${GPUS[$idx]}"
    local count="${COUNTS[$idx]}"
    local seed physical_gpu
    seed="$(seed_for_gpu "${gpu_id}")"
    physical_gpu="$(physical_gpu_for_shard "${gpu_id}")"
    (
      run_diffgui_one "${gpu_id}" "${count}" "${seed}"
    ) >"${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    log "[diffgui] launched shard=${gpu_id} physical_gpu=${physical_gpu} count=${count} seed=${seed} timesteps=${TIMESTEPS} pid=${pids[-1]}"
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      log "[diffgui] shard=${GPUS[$idx]} completed"
    else
      log "[diffgui] shard=${GPUS[$idx]} FAILED (see ${log_dir}/gpu_${GPUS[$idx]}.log)"
      failed=1
    fi
  done
  return "${failed}"
}

verify_outputs() {
  for gpu_id in "${GPUS[@]}"; do
    local log_file="${EXP}/logs/diffgui/gpu_${gpu_id}.log"
    if ! grep -q "Using DiffGui sampling timesteps override: ${TIMESTEPS} from sample.num_timesteps" "${log_file}"; then
      log "[verify] shard=${gpu_id} missing 500-step override evidence in ${log_file}"
      return 1
    fi
  done
  log "[verify] all shards confirm sample.num_timesteps=${TIMESTEPS}"
  return 0
}

write_shard_summary() {
  conda run -n dif python - "${EXP}" "${GPUS[*]}" "${COUNTS[*]}" <<'PY' | tee -a "${MASTER_LOG}"
import sys
from pathlib import Path

exp = Path(sys.argv[1])
gpus = [int(value) for value in sys.argv[2].split()]
counts = [int(value) for value in sys.argv[3].split()]
rows = []
for gpu_id, expected in zip(gpus, counts):
    log_file = exp / "logs" / "diffgui" / f"gpu_{gpu_id}.log"
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out_dir = exp / "outputs" / "diffgui" / f"gpu_{gpu_id}"
    sdf_files = sorted(out_dir.rglob("*.sdf")) if out_dir.exists() else []
    rows.append(
        {
            "shard": gpu_id,
            "expected_attempts": expected,
            "success_log_lines": text.count("Success: "),
            "reconstruction_failures": text.count("Reconstruction error encountered."),
            "sdf_files": len(sdf_files),
        }
    )
summary = exp / "metadata" / "shard_summary.csv"
summary.parent.mkdir(parents=True, exist_ok=True)
header = ["shard", "expected_attempts", "success_log_lines", "reconstruction_failures", "sdf_files"]
with summary.open("w", encoding="utf-8", newline="") as handle:
    handle.write(",".join(header) + "\n")
    for row in rows:
        handle.write(",".join(str(row[key]) for key in header) + "\n")
print(f"shard summary written to {summary}")
for row in rows:
    print(row)
PY
}

main() {
  mkdir -p "${EXP}/logs" "${EXP}/metadata" "${EXP}/configs" "${EXP}/outputs"
  touch "${MASTER_LOG}"

  log "started DiffGui epoch80 independent generation"
  log "exp=${EXP}"
  log "checkpoint=${CHECKPOINT}"
  log "timesteps=${TIMESTEPS} batch_size=${BATCH_SIZE} gui_strength=${GUI_STRENGTH}"
  log "shards=${GPUS[*]} physical_gpus=$(for gpu_id in "${GPUS[@]}"; do physical_gpu_for_shard "${gpu_id}"; done | tr '\n' ' ')"
  log "counts=${COUNTS[*]} seed_rule=seed=gpu_id+${SEED_OFFSET} seed_plan_validation=validate_seed_plan.py"
  log "precheck_only=${PRECHECK_ONLY} conda_env=diffgui validation_env=dif"

  write_configs
  write_seed_plan
  preflight

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    log "precheck completed; exiting before generation because PRECHECK_ONLY=1"
    return
  fi

  if run_parallel; then
    log "all shards finished; elapsed=$(( ($(date +%s) - START_SEC) / 60 )) minutes"
  else
    log "at least one shard failed; elapsed=$(( ($(date +%s) - START_SEC) / 60 )) minutes"
    return 1
  fi

  verify_outputs
  write_shard_summary
  log "completed at $(date -Is)"
}

main "$@"
