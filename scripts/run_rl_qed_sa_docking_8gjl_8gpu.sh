#!/bin/bash
# QED + SA + Docking(8GJL) RL 训练启动脚本 — 8 卡 DDP
# 每卡 16 分子，全局 128 分子/epoch
# 用法：
#   bash scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh
# 可选环境变量：
#   SEED=42 LOG_LEVEL=INFO GPUS=0,1,2,3,4,5,6,7 bash scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh

set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
DEFAULT_CONFIG_PATH="$PROJECT_ROOT/configs/qed_sa_docking_8gjl_8gpu.yaml"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-42}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH"
  exit 1
fi

if [ ! -f "$CONDA_SH" ]; then
  echo "[ERROR] Conda init script not found: $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[ERROR] Python not found or not executable: $PYTHON_BIN"
  exit 1
fi

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPUS"

mkdir -p "$PROJECT_ROOT/OUT"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "QED+SA+Docking(8GJL) RL 训练 — 8 卡 DDP (全局 128 分子/epoch)"
echo "============================================================"
echo "Project root : $PROJECT_ROOT"
echo "Config       : $CONFIG_PATH"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Seed         : $SEED"
echo "Start time   : $(date '+%F %T')"
echo "============================================================"

nvidia-smi --list-gpus || true

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  ddpo/main_diffsbdd_ddp.py \
    --config_path "$CONFIG_PATH" \
    --seed "$SEED" \
    --log_level "$LOG_LEVEL"

echo "============================================================"
echo "8 卡 DDP 训练完成"
echo "End time: $(date '+%F %T')"
echo "============================================================"
