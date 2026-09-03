#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${DIFFB110_PROJECT_ROOT:-/data/disk6/DiffB110}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/launch/diffsbdd_global_e2po_structure_matched_x_only_test_260901.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/analysis/diffsbdd_global_e2po_structure_matched_x_only_test_260901}"
GPU_ID="${GPU_ID:-1}"
PAIR_COUNT="${PAIR_COUNT:-4096}"
EPOCHS="${EPOCHS:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR missing config: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/analysis/diffsbdd_structure_pairs_matched_260831/train_pairs.csv" ]]; then
  echo "ERROR missing matched train pair manifest" >&2
  exit 2
fi
if [[ -f "${OUTPUT_DIR}/completion.json" ]]; then
  echo "ERROR completed output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
echo "$(date -Is) INFO launcher_pid=$$ physical_gpu=${GPU_ID} pair_count=${PAIR_COUNT} epochs=${EPOCHS}"
echo "$(date -Is) INFO config=${CONFIG_PATH} output_dir=${OUTPUT_DIR}"
echo "$(date -Is) INFO python=${PYTHON_BIN}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/DiffSBDD${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/ddpo/main_diffsbdd_global_dpo.py" \
  --config "${CONFIG_PATH}" \
  --variant x_only \
  --pair-count "${PAIR_COUNT}" \
  --epochs "${EPOCHS}" \
  --disable-generation \
  --output-dir "${OUTPUT_DIR}"
