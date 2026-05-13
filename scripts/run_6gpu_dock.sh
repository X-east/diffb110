#!/bin/bash
# QED + SA + Docking(8GJL UniDock) — 6 卡生产训练
set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
DEFAULT_CONFIG_PATH="$PROJECT_ROOT/configs/6gpu_dock.yaml"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [ ! -f "$CONFIG_PATH" ]; then echo "[ERROR] Config not found: $CONFIG_PATH"; exit 1; fi
if [ ! -f "$CONDA_SH" ];   then echo "[ERROR] Conda init script not found: $CONDA_SH"; exit 1; fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if [ ! -x "$PYTHON_BIN" ]; then echo "[ERROR] Python not found: $PYTHON_BIN"; exit 1; fi

eval "$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/yaml_launch_env.py" "$CONFIG_PATH")"

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$YAML_GPUS"

mkdir -p "$PROJECT_ROOT/OUT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "$YAML_RUN_NAME — ${YAML_NUM_GPUS} 卡 DDP"
echo "============================================================"
echo "Config : $CONFIG_PATH"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Start  : $(date '+%F %T')"
echo "============================================================"

torchrun \
  --nproc_per_node="$YAML_NUM_GPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  ddpo/main_diffsbdd_ddp.py \
    --config_path "$CONFIG_PATH" \
    --log_level "$LOG_LEVEL"

echo "============================================================"
echo "${YAML_NUM_GPUS} 卡 DDP 训练完成  End: $(date '+%F %T')"
echo "============================================================"
