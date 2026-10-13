#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui_collection/diffgui-original-t200-8gpu-trajectory-50k_20260727_212433}"
CONFIG_PATH="${CONFIG_PATH:-$OUTPUT_DIR/config/launch_config_snapshot.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29751}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/logs/resume_20260728.log}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$OUTPUT_DIR/source_snapshot_resume_20260728}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs" "${SNAPSHOT_DIR}"
cd "${ROOT}"

conda run -n "${CONDA_ENV}" python -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); g=int(sys.argv[2]); b=int(c['diffgui_collection']['batch_size_per_rank']); s=int(c['diffgui_collection']['num_shards_per_rank']); t=int(c['diffgui_docking_batch_sweep']['num_timesteps']); assert g==8 and b==10 and s==625 and t==200 and g*b*s==50000; assert c['diffgui_docking_batch_sweep']['seed_mode']=='gpu_id'; print('resume_precheck_ok attempts=',g*b*s)" "${CONFIG_PATH}" "${NPROC_PER_NODE}"
conda run -n "${CONDA_ENV}" python -m py_compile \
  ddpo/main_diffgui_collect_ddp.py \
  ddpo/diffgui/runtime/logprob_adapter.py \
  ddpo/diffgui/runtime/actor.py

cp ddpo/main_diffgui_collect_ddp.py "${SNAPSHOT_DIR}/"
cp ddpo/diffgui/runtime/logprob_adapter.py "${SNAPSHOT_DIR}/"
cp ddpo/diffgui/runtime/actor.py "${SNAPSHOT_DIR}/"
cp launch/run_diffgui_original_t200_8gpu_trajectory_50k_resume_20260728.sh "${SNAPSHOT_DIR}/"
sha256sum "${SNAPSHOT_DIR}/"*.py "${SNAPSHOT_DIR}/"*.sh >"${SNAPSHOT_DIR}/sha256sums.txt"

echo "[DiffGui-Collect-50k-Resume] output=${OUTPUT_DIR}"
echo "[DiffGui-Collect-50k-Resume] config=${CONFIG_PATH}"
echo "[DiffGui-Collect-50k-Resume] env=${CONDA_ENV} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-Collect-50k-Resume] start=$(date '+%F %T')"

conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_collect_ddp.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume \
  >"${LOG_PATH}" 2>&1

echo "[DiffGui-Collect-50k-Resume] completed=$(date '+%F %T')"

