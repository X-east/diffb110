#!/usr/bin/env bash
# Generate three additive DiffSBDD/Original repeats for the frozen ChEMBL UMAP.
# This script intentionally does not modify the original three benchmark repeats.
set -euo pipefail

ROOT="${ROOT:-/data/disk6/DiffB110}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/analysis/benchmark/benchmark5/original_additional_repeats_260826}"
SHARD_SCRIPT="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/run_diffsbdd_shard.py"
AGGREGATE_SCRIPT="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/aggregate_sdf.py"
CHECKPOINT="${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
PDBFILE="${ROOT}/data/8GJL.pdb"
UMAP_ROOT="${ROOT}/analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001"

GPUS=(1 2 3 4 5 6 7)
COUNTS=(143 143 143 143 143 143 142)
REPEATS=(4 5 6)
SEED_OFFSETS=(21 28 35)
TIMESTEPS="${TIMESTEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-10}"

MASTER_LOG="${RUN_ROOT}/run_original_repeats.log"
STATUS_FILE="${RUN_ROOT}/status.tsv"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    log "ERROR missing ${label}: ${path}"
    exit 1
  fi
}

mark_status() {
  local repeat="$1"
  local stage="$2"
  local state="$3"
  printf '%s\t%s\t%s\t%s\n' "${repeat}" "${stage}" "${state}" "$(date -Is)" >>"${STATUS_FILE}"
}

seed_for() {
  local gpu_id="$1"
  local seed_offset="$2"
  echo $((gpu_id + seed_offset))
}

prepare_repeat() {
  local repeat="$1"
  local seed_offset="$2"
  local repeat_root="${RUN_ROOT}/repeat${repeat}_seed$((seed_offset + 1))_$((seed_offset + 7))"
  mkdir -p "${repeat_root}/outputs/diffsbdd" "${repeat_root}/logs/diffsbdd" "${repeat_root}/aggregated" "${repeat_root}/metadata"

  local plan="${repeat_root}/metadata/seed_plan.csv"
  {
    echo "repeat,model,gpu_id,count,seed,timesteps,batch_size,reconstruction_method,output_dir"
    for idx in "${!GPUS[@]}"; do
      local gpu_id="${GPUS[$idx]}"
      local count="${COUNTS[$idx]}"
      local seed
      seed="$(seed_for "${gpu_id}" "${seed_offset}")"
      local output_dir="${repeat_root}/outputs/diffsbdd/gpu_${gpu_id}"
      echo "${repeat},DiffSBDD,${gpu_id},${count},${seed},${TIMESTEPS},${BATCH_SIZE},openbabel,${output_dir}"
    done
  } >"${plan}"
  cp "${plan}" "${repeat_root}/run_manifest.csv"
  printf '%s\n' "${repeat_root}"
}

run_one_shard() {
  local repeat="$1"
  local seed_offset="$2"
  local gpu_id="$3"
  local count="$4"
  local seed
  seed="$(seed_for "${gpu_id}" "${seed_offset}")"
  local repeat_root="${RUN_ROOT}/repeat${repeat}_seed$((seed_offset + 1))_$((seed_offset + 7))"
  local out_dir="${repeat_root}/outputs/diffsbdd/gpu_${gpu_id}"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/generated.sdf" ]]; then
    log "repeat=${repeat} gpu=${gpu_id} existing generated.sdf; skip"
    return 0
  fi
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export PYTHONDONTWRITEBYTECODE=1
  conda run -n dif python "${SHARD_SCRIPT}" \
    --repo-root "${ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --pdbfile "${PDBFILE}" \
    --ref-ligand "C:1101" \
    --outfile "${out_dir}/generated.sdf" \
    --n-samples "${count}" \
    --batch-size "${BATCH_SIZE}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${seed}" \
    --reconstruction-method openbabel
}

run_repeat() {
  local repeat="$1"
  local seed_offset="$2"
  local repeat_root="${RUN_ROOT}/repeat${repeat}_seed$((seed_offset + 1))_$((seed_offset + 7))"
  local shard_log_dir="${repeat_root}/logs/diffsbdd"
  local pids=()
  local failed=0

  mark_status "${repeat}" "generation" "started"
  log "repeat=${repeat} generation started; seeds=$((seed_offset + 1))-$((seed_offset + 7)); timesteps=${TIMESTEPS}"
  for idx in "${!GPUS[@]}"; do
    local gpu_id="${GPUS[$idx]}"
    local count="${COUNTS[$idx]}"
    run_one_shard "${repeat}" "${seed_offset}" "${gpu_id}" "${count}" >"${shard_log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    log "repeat=${repeat} gpu=${gpu_id} count=${count} seed=$(seed_for "${gpu_id}" "${seed_offset}") pid=${pids[-1]}"
  done
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      log "repeat=${repeat} gpu=${GPUS[$idx]} completed"
    else
      log "ERROR repeat=${repeat} gpu=${GPUS[$idx]} failed; see ${shard_log_dir}/gpu_${GPUS[$idx]}.log"
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    mark_status "${repeat}" "generation" "failed"
    return 1
  fi
  mark_status "${repeat}" "generation" "completed"

  local aggregate="${repeat_root}/aggregated/diffsbdd.sdf"
  if [[ ! -s "${aggregate}" ]]; then
    mark_status "${repeat}" "aggregation" "started"
    conda run -n dif python "${AGGREGATE_SCRIPT}" \
      --model DiffSBDD \
      --output "${aggregate}" \
      --limit 1000 \
      "${repeat_root}/outputs/diffsbdd" | tee -a "${MASTER_LOG}"
    mark_status "${repeat}" "aggregation" "completed"
  else
    log "repeat=${repeat} existing aggregate; skip"
    mark_status "${repeat}" "aggregation" "existing"
  fi
  require_file "repeat ${repeat} aggregate" "${aggregate}"
}

main() {
  mkdir -p "${RUN_ROOT}"
  : >"${MASTER_LOG}"
  printf 'repeat\tstage\tstate\ttimestamp\n' >"${STATUS_FILE}"
  log "started additive Original repeats"
  log "run_root=${RUN_ROOT}"
  log "protocol=DiffSBDD; attempts_per_repeat=1000; timesteps=${TIMESTEPS}; batch_size=${BATCH_SIZE}; reconstruction=openbabel"
  log "gpu_shards=${GPUS[*]}; counts=${COUNTS[*]}"
  log "frozen_umap=${UMAP_ROOT}/umap_model.joblib"
  require_file "DiffSBDD shard script" "${SHARD_SCRIPT}"
  require_file "DiffSBDD checkpoint" "${CHECKPOINT}"
  require_file "8GJL PDB" "${PDBFILE}"
  require_file "frozen UMAP model" "${UMAP_ROOT}/umap_model.joblib"
  require_file "frozen UMAP generated metrics" "${UMAP_ROOT}/generated_embedding_metrics.csv.gz"
  if [[ "${TIMESTEPS}" != "1000" ]]; then
    log "ERROR expected TIMESTEPS=1000, got ${TIMESTEPS}"
    exit 1
  fi

  for idx in "${!REPEATS[@]}"; do
    local repeat="${REPEATS[$idx]}"
    local seed_offset="${SEED_OFFSETS[$idx]}"
    prepare_repeat "${repeat}" "${seed_offset}" >/dev/null
    run_repeat "${repeat}" "${seed_offset}"
  done
  log "all three additive repeats completed"
}

main "$@"
