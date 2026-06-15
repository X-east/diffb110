#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"
COUNTS=(72 72 72 71 71 71 71)
GPUS=(0 1 2 3 4 5 6)
TIMESTEPS=500

mkdir -p "${EXP}/logs" "${EXP}/outputs" "${EXP}/aggregated" "${EXP}/evaluation"

run_parallel_shards() {
  local model_name="$1"
  local log_dir="${EXP}/logs/${model_name}"
  mkdir -p "${log_dir}"
  shift

  local pids=()
  local gpu_id
  for gpu_id in "${GPUS[@]}"; do
    local count="${COUNTS[${gpu_id}]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONDONTWRITEBYTECODE=1
      "$@" "${gpu_id}" "${count}"
    ) > "${log_dir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    echo "[${model_name}] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
  done

  local failed=0
  local idx
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[${model_name}] gpu=${idx} completed"
    else
      echo "[${model_name}] gpu=${idx} failed; see ${log_dir}/gpu_${idx}.log"
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
}

run_pmdm_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/pmdm/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/pmdm/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n pmdm_modern_ref python "${ROOT}/analysis/modern_sbdd_20260531/run_pmdm_for_pdb.py" \
    --repo "${ROOT}/reference project/modern_sbdd_20260531/PMDM" \
    --ckpt "${ROOT}/analysis/modern_sbdd_20260531/assets/PMDM/500.pt" \
    --pdb-path "${INPUTS}/8GJL_C1101_pocket8A.pdb" \
    --outdir "${out_dir}" \
    --num-atom 24 \
    --num-samples "${count}" \
    --batch-size 32 \
    --n-steps "${TIMESTEPS}" \
    --device cuda
}

aggregate_pmdm() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model PMDM \
    --output "${EXP}/aggregated/pmdm.sdf" \
    --limit 500 \
    "${EXP}/outputs/pmdm"
}

run_diffint_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffint/gpu_${gpu_id}"
  local work_dir="${EXP}/work/diffint/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/diffint/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${work_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n diffint_modern_ref python "${SCRIPTS}/run_interaction_model_shard.py" \
    --kind diffint \
    --repo "${ROOT}/reference project/modern_sbdd_20260531/DiffInt" \
    --checkpoint "${ROOT}/reference project/modern_sbdd_20260531/DiffInt/checkpoints/best_model.ckpt" \
    --pdb "${INPUTS}/8GJL_C1101_protein_only.pdb" \
    --sdf "${INPUTS}/8GJL_C1101_ligand.sdf" \
    --outdir "${out_dir}" \
    --workdir "${work_dir}" \
    --n-samples "${count}" \
    --batch-size 72 \
    --timesteps "${TIMESTEPS}"
}

aggregate_diffint() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffInt \
    --output "${EXP}/aggregated/diffint.sdf" \
    --limit 500 \
    "${EXP}/outputs/diffint/gpu_0/processed" \
    "${EXP}/outputs/diffint/gpu_1/processed" \
    "${EXP}/outputs/diffint/gpu_2/processed" \
    "${EXP}/outputs/diffint/gpu_3/processed" \
    "${EXP}/outputs/diffint/gpu_4/processed" \
    "${EXP}/outputs/diffint/gpu_5/processed" \
    "${EXP}/outputs/diffint/gpu_6/processed"
}

run_diffpharma_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffpharma/gpu_${gpu_id}"
  local work_dir="${EXP}/work/diffpharma/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/diffpharma/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${work_dir}" "${mpl_dir}"
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n diffpharma_modern_ref python "${SCRIPTS}/run_interaction_model_shard.py" \
    --kind diffpharma \
    --repo "${ROOT}/reference project/modern_sbdd_20260531/DiffPharma" \
    --checkpoint "${ROOT}/analysis/modern_sbdd_20260531/assets/DiffPharma/DiffPharma_best.ckpt" \
    --pdb "${INPUTS}/8GJL_C1101_protein_only.pdb" \
    --sdf "${INPUTS}/8GJL_C1101_ligand.sdf" \
    --outdir "${out_dir}" \
    --workdir "${work_dir}" \
    --n-samples "${count}" \
    --batch-size 24 \
    --timesteps "${TIMESTEPS}"
}

aggregate_diffpharma() {
  conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
    --model DiffPharma \
    --output "${EXP}/aggregated/diffpharma.sdf" \
    --limit 500 \
    "${EXP}/outputs/diffpharma/gpu_0/processed" \
    "${EXP}/outputs/diffpharma/gpu_1/processed" \
    "${EXP}/outputs/diffpharma/gpu_2/processed" \
    "${EXP}/outputs/diffpharma/gpu_3/processed" \
    "${EXP}/outputs/diffpharma/gpu_4/processed" \
    "${EXP}/outputs/diffpharma/gpu_5/processed" \
    "${EXP}/outputs/diffpharma/gpu_6/processed"
}

evaluate_all_available() {
  conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
    --out-dir "${EXP}/evaluation/six_models" \
    --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
    --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
    --sdf "DePPA=${EXP}/aggregated/deppa.sdf" \
    --sdf "PMDM=${EXP}/aggregated/pmdm.sdf" \
    --sdf "DiffInt=${EXP}/aggregated/diffint.sdf" \
    --sdf "DiffPharma=${EXP}/aggregated/diffpharma.sdf"
}

echo "[second_benchmark] started at $(date -Is)"

run_parallel_shards pmdm run_pmdm_one
aggregate_pmdm

run_parallel_shards diffint run_diffint_one
aggregate_diffint

run_parallel_shards diffpharma run_diffpharma_one
aggregate_diffpharma

evaluate_all_available

echo "[second_benchmark] finished at $(date -Is)"
