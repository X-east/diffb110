#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_original_t200_8gpu_trajectory_50k_20260727.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29750}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui_collection/diffgui-original-t200-8gpu-trajectory-50k_${RUN_STAMP}}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/logs/launch.log}"
RESUME="${RESUME:-0}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/config" "${OUTPUT_DIR}/source_snapshot"
cd "${ROOT}"

conda run -n "${CONDA_ENV}" python -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); g=int(sys.argv[2]); b=int(c['diffgui_collection']['batch_size_per_rank']); s=int(c['diffgui_collection']['num_shards_per_rank']); t=int(c['diffgui_docking_batch_sweep']['num_timesteps']); assert g==8 and b==10 and s==625 and t==200; assert g*b*s==50000; assert c['diffgui_docking_batch_sweep']['seed_mode']=='gpu_id'; print('precheck_ok attempts=',g*b*s)" "${CONFIG_PATH}" "${NPROC_PER_NODE}"
conda run -n "${CONDA_ENV}" python -m py_compile \
  ddpo/main_diffgui_collect_ddp.py \
  ddpo/diffgui/runtime/logprob_adapter.py \
  ddpo/diffgui/runtime/actor.py

cp "${CONFIG_PATH}" "${OUTPUT_DIR}/config/launch_config_snapshot.yaml"
cp ddpo/main_diffgui_collect_ddp.py "${OUTPUT_DIR}/source_snapshot/"
cp ddpo/diffgui/runtime/logprob_adapter.py "${OUTPUT_DIR}/source_snapshot/"
cp ddpo/diffgui/runtime/actor.py "${OUTPUT_DIR}/source_snapshot/"
cp launch/run_diffgui_original_t200_8gpu_trajectory_50k_20260727.sh "${OUTPUT_DIR}/source_snapshot/"
sha256sum \
  "reference project/代码/DiffGui/ckpt/trained.pt" \
  "${OUTPUT_DIR}/config/launch_config_snapshot.yaml" \
  "${OUTPUT_DIR}/source_snapshot/"*.py \
  "${OUTPUT_DIR}/source_snapshot/"*.sh \
  >"${OUTPUT_DIR}/source_snapshot/sha256sums.txt"

echo "[DiffGui-Collect-50k] config_snapshot=${OUTPUT_DIR}/config/launch_config_snapshot.yaml"
echo "[DiffGui-Collect-50k] output=${OUTPUT_DIR}"
echo "[DiffGui-Collect-50k] env=${CONDA_ENV} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[DiffGui-Collect-50k] attempts=50000 ranks=8 attempts_per_rank=6250 batch_per_rank=10 shards_per_rank=625 steps=200"
echo "[DiffGui-Collect-50k] resume=${RESUME} start=$(date '+%F %T')"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

conda run -n "${CONDA_ENV}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ddpo/main_diffgui_collect_ddp.py \
  --config "${OUTPUT_DIR}/config/launch_config_snapshot.yaml" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  >"${LOG_PATH}" 2>&1

echo "[DiffGui-Collect-50k] completed=$(date '+%F %T')"

