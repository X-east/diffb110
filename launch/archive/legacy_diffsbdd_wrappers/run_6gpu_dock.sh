#!/bin/bash
# QED + SA + Docking(8GJL UniDock) — 6 卡生产训练
set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"
DEFAULT_CONFIG_PATH="$PROJECT_ROOT/launch/6gpu_dock.yaml"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CHEMBL_PRIOR_ENABLE="${CHEMBL_PRIOR_ENABLE:-1}"
CHEMBL_PRIOR_WEIGHT="${CHEMBL_PRIOR_WEIGHT:-0.1}"
CHEMBL_PRIOR_CACHE="${CHEMBL_PRIOR_CACHE:-./data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz}"
CHEMBL_PRIOR_TOP_K="${CHEMBL_PRIOR_TOP_K:-10}"
CHEMBL_PRIOR_CHUNK_SIZE="${CHEMBL_PRIOR_CHUNK_SIZE:-100000}"
CHEMBL_PRIOR_LOW_THRESHOLD="${CHEMBL_PRIOR_LOW_THRESHOLD:-0.25}"
CHEMBL_PRIOR_LOW_WEIGHT="${CHEMBL_PRIOR_LOW_WEIGHT:-4.0}"
CHEMBL_PRIOR_HIGH_THRESHOLD="${CHEMBL_PRIOR_HIGH_THRESHOLD:-1.0}"
CHEMBL_PRIOR_HIGH_WEIGHT="${CHEMBL_PRIOR_HIGH_WEIGHT:-0.0}"
CHEMBL_PRIOR_FAILURE_PENALTY="${CHEMBL_PRIOR_FAILURE_PENALTY:--2.0}"
CHEMBL_PRIOR_MAX_REWARD_SIM="${CHEMBL_PRIOR_MAX_REWARD_SIM:-0.50}"

if [ ! -f "$CONFIG_PATH" ]; then echo "[ERROR] Config not found: $CONFIG_PATH"; exit 1; fi
if [ ! -f "$CONDA_SH" ];   then echo "[ERROR] Conda init script not found: $CONDA_SH"; exit 1; fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if [ ! -x "$PYTHON_BIN" ]; then echo "[ERROR] Python not found: $PYTHON_BIN"; exit 1; fi

GENERATED_CONFIG_DIR="$PROJECT_ROOT/launch/generated/chembl_prior"
mkdir -p "$GENERATED_CONFIG_DIR"
export CONFIG_PATH CHEMBL_PRIOR_ENABLE CHEMBL_PRIOR_WEIGHT CHEMBL_PRIOR_CACHE \
  CHEMBL_PRIOR_TOP_K CHEMBL_PRIOR_CHUNK_SIZE CHEMBL_PRIOR_LOW_THRESHOLD \
  CHEMBL_PRIOR_LOW_WEIGHT CHEMBL_PRIOR_HIGH_THRESHOLD CHEMBL_PRIOR_HIGH_WEIGHT \
  CHEMBL_PRIOR_FAILURE_PENALTY CHEMBL_PRIOR_MAX_REWARD_SIM GENERATED_CONFIG_DIR

CONFIG_PATH="$($PYTHON_BIN << 'PY'
import os
import time
from pathlib import Path
import yaml

base = Path(os.environ["CONFIG_PATH"]).expanduser().resolve()
out_dir = Path(os.environ["GENERATED_CONFIG_DIR"]).expanduser().resolve()
data = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
reward = data.setdefault("reward", {})
weights = reward.setdefault("weights", {})
chembl = reward.setdefault("chembl_prior", {})

# CHEMBL_PRIOR_ENABLE=0 → 关掉权重
disable = os.environ.get("CHEMBL_PRIOR_ENABLE", "1")
if disable in {"0", "false", "False", "no", "NO"}:
    weights["chembl_prior"] = 0.0
else:
    # YAML 是数据源，缺失时才 fallback 到环境变量
    weights.setdefault("chembl_prior", float(os.environ.get("CHEMBL_PRIOR_WEIGHT", 0.1)))
    chembl.setdefault("fingerprint_cache",
                      os.environ.get("CHEMBL_PRIOR_CACHE", "./data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"))
    chembl.setdefault("top_k", int(os.environ.get("CHEMBL_PRIOR_TOP_K", 5)))
    chembl.setdefault("chunk_size", int(os.environ.get("CHEMBL_PRIOR_CHUNK_SIZE", 100000)))
    chembl.setdefault("low_similarity_threshold", float(os.environ.get("CHEMBL_PRIOR_LOW_THRESHOLD", 0.25)))
    chembl.setdefault("low_similarity_weight", float(os.environ.get("CHEMBL_PRIOR_LOW_WEIGHT", 4.0)))
    chembl.setdefault("high_similarity_threshold", float(os.environ.get("CHEMBL_PRIOR_HIGH_THRESHOLD", 0.85)))
    chembl.setdefault("high_similarity_weight", float(os.environ.get("CHEMBL_PRIOR_HIGH_WEIGHT", 0.1)))
    chembl.setdefault("failure_penalty", float(os.environ.get("CHEMBL_PRIOR_FAILURE_PENALTY", "-1.0")))
    chembl.setdefault("max_reward_similarity", float(os.environ.get("CHEMBL_PRIOR_MAX_REWARD_SIM", "0.50")))

if weights.get("chembl_prior", 0.0):
    data["name"] = f"{data.get('name', '6gpu-dock')}-chembl"

out_dir.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
out = out_dir / f"{base.stem}_chembl_{stamp}.yaml"
out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(out)
PY
)"

eval "$("$PYTHON_BIN" "$PROJECT_ROOT/launch/yaml_launch_env.py" "$CONFIG_PATH")"

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
echo "ChEMBL : enable=$CHEMBL_PRIOR_ENABLE weight=$CHEMBL_PRIOR_WEIGHT low_thr=$CHEMBL_PRIOR_LOW_THRESHOLD high_thr=$CHEMBL_PRIOR_HIGH_THRESHOLD cache=$CHEMBL_PRIOR_CACHE"
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
