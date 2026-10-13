#!/usr/bin/env bash
# Recover only missing shards from additive Original repeat 6.
set -uo pipefail

ROOT="${ROOT:-/data/disk6/DiffB110}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/analysis/benchmark/benchmark5/original_additional_repeats_260826}"
REPEAT_ROOT="${RUN_ROOT}/repeat6_seed36_42"
SHARD_SCRIPT="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/run_diffsbdd_shard.py"
AGGREGATE_SCRIPT="${ROOT}/analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/aggregate_sdf.py"
CHECKPOINT="${ROOT}/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
PDBFILE="${ROOT}/data/8GJL.pdb"
GPUS=(1 2 3 4 5 6 7)
COUNTS=(143 143 143 143 143 143 142)
SEEDS=(36 37 38 39 40 41 42)

LOG="${REPEAT_ROOT}/recovery.log"
mkdir -p "${REPEAT_ROOT}/logs/diffsbdd" "${REPEAT_ROOT}/aggregated"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] repeat6 recovery started"
echo "[$(date -Is)] existing GPU3 output is retained; missing shards will be regenerated"

pids=()
active=()
for idx in "${!GPUS[@]}"; do
  gpu_id="${GPUS[$idx]}"
  count="${COUNTS[$idx]}"
  seed="${SEEDS[$idx]}"
  out_dir="${REPEAT_ROOT}/outputs/diffsbdd/gpu_${gpu_id}"
  out_file="${out_dir}/generated.sdf"
  mkdir -p "${out_dir}"
  if [[ -s "${out_file}" ]]; then
    echo "[$(date -Is)] gpu=${gpu_id} seed=${seed} existing output; skip"
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTHONDONTWRITEBYTECODE=1
    conda run -n dif python "${SHARD_SCRIPT}" \
      --repo-root "${ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --pdbfile "${PDBFILE}" \
      --ref-ligand "C:1101" \
      --outfile "${out_file}" \
      --n-samples "${count}" \
      --batch-size 10 \
      --timesteps 1000 \
      --seed "${seed}" \
      --reconstruction-method openbabel
  ) >"${REPEAT_ROOT}/logs/diffsbdd/gpu_${gpu_id}.recovery.log" 2>&1 &
  pid="$!"
  pids+=("${pid}")
  active+=("${gpu_id}")
  echo "[$(date -Is)] gpu=${gpu_id} count=${count} seed=${seed} pid=${pid} launched"
done

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[$(date -Is)] gpu=${active[$idx]} completed"
  else
    echo "[$(date -Is)] ERROR gpu=${active[$idx]} failed; see ${REPEAT_ROOT}/logs/diffsbdd/gpu_${active[$idx]}.recovery.log"
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "[$(date -Is)] recovery failed; aggregation not attempted"
  exit 1
fi

missing=0
for gpu_id in "${GPUS[@]}"; do
  out_file="${REPEAT_ROOT}/outputs/diffsbdd/gpu_${gpu_id}/generated.sdf"
  if [[ ! -s "${out_file}" ]]; then
    echo "[$(date -Is)] ERROR missing ${out_file}"
    missing=1
  fi
done
if [[ "${missing}" != "0" ]]; then
  echo "[$(date -Is)] recovery incomplete; aggregation not attempted"
  exit 1
fi

conda run -n dif python "${AGGREGATE_SCRIPT}" \
  --model DiffSBDD \
  --output "${REPEAT_ROOT}/aggregated/diffsbdd.sdf" \
  --limit 1000 \
  "${REPEAT_ROOT}/outputs/diffsbdd" \
  | tee -a "${LOG}"
echo "[$(date -Is)] repeat6 recovery and aggregation completed"
