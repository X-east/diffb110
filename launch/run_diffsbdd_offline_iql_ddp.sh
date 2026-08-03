#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
GIT_COMMON_DIR="$(git -C "${PROJECT_ROOT}" rev-parse --path-format=absolute --git-common-dir)"
DEFAULT_DATA_ROOT="$(dirname "${GIT_COMMON_DIR}")"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/launch/diffsbdd_offline_iql_ddp_smoke_260731.yaml}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/analysis/offline_iql_ddp_smoke_260731}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29631}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
EPOCHS="${EPOCHS:-}"
RESUME_MANIFEST="${RESUME_MANIFEST:-}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ "${NPROC_PER_NODE}" -le 0 ]]; then
  echo "NPROC_PER_NODE must be positive" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
LAUNCH_LOG="${OUTPUT_DIR}/launch.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "$(date --iso-8601=seconds) launch start"
echo "config=${CONFIG_PATH}"
echo "data_root=${DATA_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "conda_env=dif"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all-visible}"
echo "world_size=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
echo "git_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
git -C "${PROJECT_ROOT}" status --short
conda run -n dif python -c \
  'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} visible_gpus={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

COMMAND=(
  conda run -n dif torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  "${PROJECT_ROOT}/ddpo/main_diffsbdd_offline_iql_ddp.py"
  --config "${CONFIG_PATH}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --log-level "${LOG_LEVEL}"
)
if [[ -n "${EPOCHS}" ]]; then
  COMMAND+=(--epochs "${EPOCHS}")
fi
if [[ -n "${RESUME_MANIFEST}" ]]; then
  COMMAND+=(--resume-manifest "${RESUME_MANIFEST}")
fi
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  COMMAND+=(--preflight-only)
fi

"${COMMAND[@]}"
echo "$(date --iso-8601=seconds) launch completed"
