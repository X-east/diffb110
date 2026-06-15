#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/disk6/DiffB110/analysis/diffgui_6gpu_500x3_20260531_085146"
PROJECT="/data/disk6/DiffB110/reference project/DiffGui"
PATCH_DIR="${ROOT}/runtime_patch"
COUNTS=(84 84 83 83 83 83)
STEPS=(100 200 1000)
BATCH_SIZE=2

cd "${PROJECT}"
conda run -n diffgui python "${ROOT}/prepare_diffgui_configs.py"

for steps in "${STEPS[@]}"; do
  echo "[launch] starting diffusion steps=${steps}"
  pids=()
  for gpu_id in 0 1 2 3 4 5; do
    cfg="${ROOT}/configs/diffgui_steps${steps}_gpu${gpu_id}.yml"
    outdir="${ROOT}/outputs/steps_${steps}/gpu_${gpu_id}"
    logdir="${ROOT}/logs/steps_${steps}"
    mpldir="${ROOT}/mplconfig/steps_${steps}/gpu_${gpu_id}"
    mkdir -p "${outdir}" "${logdir}" "${mpldir}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export DIFFGUI_NUM_TIMESTEPS="${steps}"
      export PYTHONPATH="${PATCH_DIR}:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
      export PYTHONDONTWRITEBYTECODE=1
      export MPLCONFIGDIR="${mpldir}"
      export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
      /usr/bin/time -p conda run -n diffgui python scripts/sample.py \
        --config "${cfg}" \
        --outdir "${outdir}" \
        --device cuda:0 \
        --batch_size "${BATCH_SIZE}"
    ) > "${logdir}/gpu_${gpu_id}.log" 2>&1 &
    pids+=("$!")
    echo "[launch] steps=${steps} gpu=${gpu_id} pid=${pids[-1]} target=${COUNTS[$gpu_id]}"
  done

  failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[launch] steps=${steps} gpu=${idx} completed"
    else
      echo "[launch] steps=${steps} gpu=${idx} failed; see ${ROOT}/logs/steps_${steps}/gpu_${idx}.log"
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi

  conda run -n diffgui python "${ROOT}/aggregate_diffgui_batch.py" \
    --root "${ROOT}" \
    --steps "${steps}" \
    --counts "84,84,83,83,83,83" \
    > "${ROOT}/logs/aggregate_steps_${steps}.log" 2>&1
  echo "[launch] aggregated steps=${steps}"
done

echo "[launch] all DiffGui batches finished"
