#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/md666/data/disk6/DiffB110"
EXP="${ROOT}/analysis/8gjl_c1101_benchmark_20260603_500mol_500steps"
INPUTS="${EXP}/inputs"
SCRIPTS="${EXP}/scripts"
COUNTS=(72 72 72 71 71 71 71)
GPUS=(0 1 2 3 4 5 6)
TIMESTEPS=500

mkdir -p "${EXP}/logs/deppa_rerun" "${EXP}/outputs/deppa_rerun" "${EXP}/aggregated"

run_deppa_one() {
  local gpu_id="$1"
  local count="$2"
  local project="${ROOT}/reference project/DePPA"
  local out_dir="${EXP}/outputs/deppa_rerun/gpu_${gpu_id}"
  local mpl_dir="${EXP}/mplconfig/deppa/gpu_${gpu_id}"
  mkdir -p "${mpl_dir}"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONDONTWRITEBYTECODE=1
  export MPLCONFIGDIR="${mpl_dir}"
  /usr/bin/time -p conda run -n deppa_ref python "${project}/test.py" \
    "${ROOT}/analysis/reference_project_generation_20260530_173617/deppa_downloads/crossdocked_fullatom_cond.ckpt" \
    --test_dir "${INPUTS}/deppa_test_pocket" \
    --outdir "${out_dir}" \
    --n_samples "${count}" \
    --batch_size 10 \
    --timesteps "${TIMESTEPS}"
}

echo "[deppa_continue] started at $(date -Is)"
pids=()
for gpu_id in "${GPUS[@]}"; do
  count="${COUNTS[${gpu_id}]}"
  (
    run_deppa_one "${gpu_id}" "${count}"
  ) > "${EXP}/logs/deppa_rerun/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
  echo "[deppa_continue] launched gpu=${gpu_id} count=${count} pid=${pids[-1]}"
done

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[deppa_continue] gpu=${idx} completed"
  else
    echo "[deppa_continue] gpu=${idx} failed; see ${EXP}/logs/deppa_rerun/gpu_${idx}.log"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

conda run -n dif python "${SCRIPTS}/aggregate_sdf.py" \
  --model DePPA \
  --output "${EXP}/aggregated/deppa.sdf" \
  --limit 500 \
  "${EXP}/outputs/deppa_rerun"

conda run -n dif python "${SCRIPTS}/evaluate_sdf_panel.py" \
  --out-dir "${EXP}/evaluation/core" \
  --sdf "DiffSBDD=${EXP}/aggregated/diffsbdd.sdf" \
  --sdf "DiffGui=${EXP}/aggregated/diffgui.sdf" \
  --sdf "DePPA=${EXP}/aggregated/deppa.sdf"

echo "[deppa_continue] finished at $(date -Is)"
