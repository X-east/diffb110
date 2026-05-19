#!/bin/bash
# QED + SA — 2 卡
set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
DEFAULT_CONFIG_PATH="$PROJECT_ROOT/launch/2gpu_qed_sa.yaml"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

[ ! -f "$CONFIG_PATH" ] && echo "[ERROR] Config not found: $CONFIG_PATH" && exit 1
[ ! -f "$CONDA_SH" ]   && echo "[ERROR] Conda not found: $CONDA_SH" && exit 1

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
[ ! -x "$PYTHON_BIN" ] && echo "[ERROR] Python not found: $PYTHON_BIN" && exit 1

eval "$("$PYTHON_BIN" "$PROJECT_ROOT/launch/yaml_launch_env.py" "$CONFIG_PATH")"

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$YAML_GPUS"

mkdir -p "$PROJECT_ROOT/OUT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "$YAML_RUN_NAME — ${YAML_NUM_GPUS} 卡"
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
echo "${YAML_NUM_GPUS} 卡训练完成  End: $(date '+%F %T')"
echo "============================================================"
