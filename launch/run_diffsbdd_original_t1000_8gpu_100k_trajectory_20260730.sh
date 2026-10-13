#!/usr/bin/env bash
set -euo pipefail

ROOT="${DIFFB110_PROJECT_ROOT:-/home/md666/data/disk6/DiffB110}"
RUN_DIR="${1:?Usage: $0 RUN_DIR}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT}/launch/diffsbdd_original_t1000_8gpu_100k_trajectory_20260730.yaml}"
ENTRYPOINT="${ENTRYPOINT:-${ROOT}/ddpo/main_diffsbdd_collect_ddp.py}"
VALIDATOR="${VALIDATOR:-${ROOT}/scripts/validate_diffsbdd_trajectory_run.py}"
MASTER_PORT="${MASTER_PORT:-29503}"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${RUN_DIR}/logs/launch.log") 2>&1
exec 9> "${RUN_DIR}/launch.lock"
if ! flock -n 9; then
  echo "$(date -Is) ERROR another launcher holds ${RUN_DIR}/launch.lock" >&2
  exit 1
fi

TORCHRUN_PID=""
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ "${TORCHRUN_PID}" =~ ^[0-9]+$ ]] && kill -0 "${TORCHRUN_PID}" 2>/dev/null; then
    kill -TERM -- "-${TORCHRUN_PID}" 2>/dev/null || true
    wait "${TORCHRUN_PID}" 2>/dev/null || true
  fi
  if [[ "${exit_code}" -eq 0 ]]; then
    echo "$(date -Is) INFO launcher_status=completed"
  else
    echo "$(date -Is) WARNING launcher_status=interrupted exit_code=${exit_code}"
  fi
  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "$(date -Is) INFO launch target=8GJL/C1101 fixed_attempts=100000 timesteps=1000 max_n_nodes=30 seeds=1-8"
echo "$(date -Is) INFO config=${CONFIG_PATH} entrypoint=${ENTRYPOINT} run_dir=${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DIFFB110_PROJECT_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print('cuda_preflight=PASS devices=', torch.cuda.device_count())"

setsid "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  "${ENTRYPOINT}" \
    --config "${CONFIG_PATH}" \
    --output-dir "${RUN_DIR}" \
    --resume \
    --log-level INFO &
TORCHRUN_PID="$!"
wait "${TORCHRUN_PID}"
TORCHRUN_PID=""

"${PYTHON_BIN}" "${VALIDATOR}" \
  --run-dir "${RUN_DIR}" \
  --expected-ranks 8 \
  --expected-shards-per-rank 1250 \
  --attempts-per-shard 10 \
  --timesteps 1000 \
  --max-n-nodes 30 \
  > "${RUN_DIR}/logs/completion_validation.log" 2>&1

echo "$(date -Is) INFO run complete summary=${RUN_DIR}/summary.json completion=${RUN_DIR}/completion_report.json"
