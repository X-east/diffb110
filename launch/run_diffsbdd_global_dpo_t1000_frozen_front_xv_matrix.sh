#!/usr/bin/env bash
set -euo pipefail

ROOT="${DIFFB110_PROJECT_ROOT:-/data/disk6/DiffB110}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/launch/diffsbdd_global_dpo_t1000_frozen_front_xv.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/analysis/diffsbdd_global_dpo_t1000_frozen_front_xv}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
VARIANT_LIST="${VARIANT_LIST:-x_only x_dominant balanced v_dominant v_only}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

read -r -a GPU_IDS <<< "${GPU_LIST}"
read -r -a VARIANTS <<< "${VARIANT_LIST}"
if [[ "${#GPU_IDS[@]}" -ne "${#VARIANTS[@]}" ]]; then
  echo "ERROR GPU count=${#GPU_IDS[@]} must equal variant count=${#VARIANTS[@]}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
export DIFFB110_PROJECT_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/DiffSBDD${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec > >(tee -a "${OUTPUT_ROOT}/matrix.log") 2>&1

echo "$(date -Is) INFO matrix_start config=${CONFIG_PATH} output_root=${OUTPUT_ROOT}"
echo "$(date -Is) INFO variants=${VARIANT_LIST} gpus=${GPU_LIST}"
cp "${CONFIG_PATH}" "${OUTPUT_ROOT}/config.matrix_source.yaml"

PIDS=()
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  gpu="${GPU_IDS[$index]}"
  output_dir="${OUTPUT_ROOT}/${variant}"
  mkdir -p "${output_dir}"
  cp "${CONFIG_PATH}" "${output_dir}/config.matrix_source.yaml"

  if [[ -f "${output_dir}/completion.json" && "${RESUME_COMPLETED:-0}" != "1" ]]; then
    echo "$(date -Is) INFO skip_completed variant=${variant} output_dir=${output_dir}"
    continue
  fi

  resume_args=()
  if [[ "${RESUME:-0}" == "1" && -f "${output_dir}/checkpoint_latest.json" ]]; then
    resume_args+=(--resume)
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "$(date -Is) INFO variant=${variant} physical_gpu=${gpu} output_dir=${output_dir}"
    "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(); print('cuda_preflight=PASS visible=', torch.cuda.device_count(), 'name=', torch.cuda.get_device_name(0))"
    exec "${PYTHON_BIN}" "${ROOT}/ddpo/main_diffsbdd_global_dpo.py" \
      --config "${CONFIG_PATH}" \
      --variant "${variant}" \
      --output-dir "${output_dir}" \
      "${resume_args[@]}"
  ) > >(tee -a "${output_dir}/launcher.log") 2>&1 &
  pid="$!"
  printf '%s\n' "${pid}" > "${output_dir}/launcher.pid"
  printf '%s\n' "${gpu}" > "${output_dir}/physical_gpu.txt"
  PIDS+=("${pid}")
  echo "$(date -Is) INFO launched variant=${variant} gpu=${gpu} pid=${pid}"
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  echo "$(date -Is) INFO matrix_status=completed"
else
  echo "$(date -Is) ERROR matrix_status=failed"
fi
exit "${status}"
