#!/bin/bash
# QED + SA RL 训练启动脚本 — 2 卡
# 每卡 8 分子，全局 16 分子/epoch
# 用法：bash scripts/run_rl_qed_sa_2gpu.sh

set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
CONFIG_PATH="$PROJECT_ROOT/configs/qed_sa_2gpu.yaml"

GPUS="${GPUS:-0,1}"
SEED="${SEED:-42}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

[ ! -f "$CONFIG_PATH" ] && echo "[ERROR] Config not found: $CONFIG_PATH" && exit 1
[ ! -f "$CONDA_SH" ]   && echo "[ERROR] Conda not found: $CONDA_SH" && exit 1

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

[ ! -x "$PYTHON_BIN" ] && echo "[ERROR] Python not found: $PYTHON_BIN" && exit 1

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPUS"

mkdir -p "$PROJECT_ROOT/OUT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "QED+SA RL 训练 — 2 卡 (全局 16 分子/epoch)"
echo "Config : $CONFIG_PATH"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Seed   : $SEED"
echo "Start  : $(date '+%F %T')"
echo "============================================================"
nvidia-smi --list-gpus || true

"$PYTHON_BIN" ddpo/main_diffsbdd_lightning.py \
  --config_path "$CONFIG_PATH" \
  --seed "$SEED" \
  --log_level "$LOG_LEVEL"

echo "============================================================"
echo "2 卡训练完成  End: $(date '+%F %T')"
echo "============================================================"
