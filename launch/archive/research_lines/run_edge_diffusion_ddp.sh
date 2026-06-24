#!/bin/bash
# Standalone edge diffusion DDP launch script. It does not enter the DDPO/RL path.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

CONFIG_PATH="${1:-$PROJECT_ROOT/launch/edge_diffusion_10k_6gpu.yaml}"
MASTER_PORT="${EDGE_DDP_MASTER_PORT:-29511}"

[ ! -f "$CONFIG_PATH" ] && echo "[EDGE DDP ERROR] Config not found: $CONFIG_PATH" && exit 1
[ ! -f "$CONDA_SH" ] && echo "[EDGE DDP ERROR] Conda init script not found: $CONDA_SH" && exit 1

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
[ ! -x "$PYTHON_BIN" ] && echo "[EDGE DDP ERROR] Python not found: $PYTHON_BIN" && exit 1

eval "$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/yaml_launch_env.py" "$CONFIG_PATH")"

export CUDA_VISIBLE_DEVICES="$YAML_GPUS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/DiffSBDD:$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "$YAML_RUN_NAME — standalone edge diffusion ${YAML_NUM_GPUS} GPU DDP"
echo "Config : $CONFIG_PATH"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Batch  : per_rank=$YAML_PER_RANK_BATCH_SIZE global=$YAML_GLOBAL_BATCH_SIZE"
echo "Seed   : $YAML_SEED"
echo "Port   : $MASTER_PORT"
echo "Start  : $(date '+%F %T')"
echo "============================================================"

torchrun \
  --nproc_per_node="$YAML_NUM_GPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="$MASTER_PORT" \
  DiffSBDD/analysis/train_edge_diffusion_ddp.py \
    --config "$CONFIG_PATH"

echo "============================================================"
echo "Standalone edge diffusion DDP finished  End: $(date '+%F %T')"
echo "============================================================"
