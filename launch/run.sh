#!/bin/bash
# 通用启动脚本 — 所有配置从 YAML 读取，无硬编码
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

# 配置路径：默认 run.yaml，可通过 CONFIG_PATH 环境变量覆盖
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/launch/run.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# 前置检查
[ ! -f "$CONFIG_PATH" ] && echo "[ERROR] Config not found: $CONFIG_PATH" && exit 1
[ ! -f "$CONDA_SH" ]    && echo "[ERROR] Conda init script not found: $CONDA_SH" && exit 1

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
[ ! -x "$PYTHON_BIN" ]  && echo "[ERROR] Python not found: $PYTHON_BIN" && exit 1

# 从 YAML 提取运行参数（YAML_RUN_NAME, YAML_NUM_GPUS, YAML_GPUS, ...）
eval "$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/yaml_launch_env.py" "$CONFIG_PATH")"

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export CUDA_VISIBLE_DEVICES="$YAML_GPUS"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_ROOT/OUT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "$YAML_RUN_NAME — ${YAML_NUM_GPUS} 卡 DDP"
echo "Config : $CONFIG_PATH"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Batch  : per_rank=$YAML_PER_RANK_BATCH_SIZE global=$YAML_GLOBAL_BATCH_SIZE"
echo "Seed   : $YAML_SEED"
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
