#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"
COUNTS=(72 72 72 71 71 71 71)
GPUS=(0 1 2 3 4 5 6)
TIMESTEPS=500

mkdir -p "${EXP}/logs/diffpharma" "${EXP}/outputs/diffpharma" "${EXP}/work/diffpharma" "${EXP}/aggregated"

run_diffpharma_one() {
  local gpu_id="$1"
  local count="$2"
  local out_dir="${EXP}/outputs/diffpharma/gpu_${gpu_id}"
  local work_dir="${EXP}/work/diffpharma/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/diffpharma/gpu_${gpu_id}"
  mkdir -p "${out_dir}" "${work_dir}" "${mpl_dir}"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export PYTHONDONTWRITEBYTECODE=1
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

echo "[diffpharma_continue] started at $(date -Is)"
pids=()
for gpu_id in "${GPUS[@]}"; do
  count="${COUNTS[${gpu_id}]}"
  (
    run_diffpharma_one "${gpu_id}" "${count}"
  ) > "${EXP}/logs/diffpharma/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
  echo "[diffpharma_continue] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
done

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[diffpharma_continue] gpu=${idx} completed"
  else
    echo "[diffpharma_continue] gpu=${idx} failed; see ${EXP}/logs/diffpharma/gpu_${idx}.log"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

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

conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
  --out-dir "${EXP}/evaluation/six_models" \
  --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
  --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
  --sdf "DePPA=${EXP}/aggregated/deppa.sdf" \
  --sdf "PMDM=${EXP}/aggregated/pmdm.sdf" \
  --sdf "DiffInt=${EXP}/aggregated/diffint.sdf" \
  --sdf "DiffPharma=${EXP}/aggregated/diffpharma.sdf"

echo "[diffpharma_continue] finished at $(date -Is)"
