#!/usr/bin/env bash
set -euo pipefail

ROOT="${DIFFB110_PROJECT_ROOT:-/data/disk6/DiffB110}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/launch/diffsbdd_gaussian_dpo_t1000.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/analysis/diffsbdd_gaussian_dpo_t1000_v1/formal_run}"
PYTHON_BIN="${PYTHON_BIN:-/home/md666/miniforge3/envs/dif/bin/python}"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

mkdir -p "${OUTPUT_DIR}"
export DIFFB110_PROJECT_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/DiffSBDD${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "${OUTPUT_DIR}/launcher.log") 2>&1
echo "$(date -Is) INFO config=${CONFIG_PATH} output_dir=${OUTPUT_DIR}"
echo "$(date -Is) INFO python=${PYTHON_BIN} conda_environment=${CONDA_DEFAULT_ENV:-unknown}"
"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(); print('cuda_preflight=PASS devices=', torch.cuda.device_count(), 'torch=', torch.__version__)"

setsid "${PYTHON_BIN}" "${ROOT}/ddpo/main_diffsbdd_gaussian_dpo.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" &
CHILD_PID="$!"
echo "$(date -Is) INFO child_pid=${CHILD_PID}"
wait "${CHILD_PID}"
echo "$(date -Is) INFO launcher_status=completed"
