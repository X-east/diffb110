#!/bin/bash
# KL 系数扫描 — 基于 6gpu_dock 配置
set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
BASE_CONFIG="$PROJECT_ROOT/launch/6gpu_dock.yaml"
LAUNCH_SCRIPT="$PROJECT_ROOT/launch/run_6gpu_dock.sh"
TMP_CONFIG_DIR="$PROJECT_ROOT/launch/generated/kl_sweep"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
KL_VALUES="${KL_VALUES:-0 0.01 0.02 0.05 0.1 0.2 0.5 1}"

[ ! -f "$BASE_CONFIG" ]   && echo "[ERROR] Base config not found: $BASE_CONFIG" && exit 1
[ ! -f "$LAUNCH_SCRIPT" ] && echo "[ERROR] Launch script not found: $LAUNCH_SCRIPT" && exit 1
[ ! -x "$PYTHON_BIN" ]    && echo "[ERROR] Python not found: $PYTHON_BIN" && exit 1

mkdir -p "$TMP_CONFIG_DIR"
export PROJECT_ROOT BASE_CONFIG TMP_CONFIG_DIR

printf '============================================================\n'
printf 'KL sweep: %s\n' "$KL_VALUES"
printf 'Base config : %s\n' "$BASE_CONFIG"
printf 'Launch script: %s\n' "$LAUNCH_SCRIPT"
printf '============================================================\n'

for KL in $KL_VALUES; do
  export KL
  KL_TAG=$("$PYTHON_BIN" -c "import os; print(os.environ['KL'].replace('.', 'p'))")
  export KL_TAG

  CONFIG_PATH=$("$PYTHON_BIN" -c "
import os, yaml
from pathlib import Path

base = Path(os.environ['BASE_CONFIG'])
out_dir = Path(os.environ['TMP_CONFIG_DIR'])
kl = float(os.environ['KL'])
tag = os.environ['KL_TAG']

data = yaml.safe_load(base.read_text(encoding='utf-8'))
data['name'] = f\"{data.get('name', '6gpu-dock')}-kl-{tag}\"
data.setdefault('train', {})['kl_coef'] = kl

p = out_dir / f'6gpu_dock_kl_{tag}.yaml'
p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
print(p)
")

  printf '\n[KL SWEEP] kl_coef=%s  Config: %s\n' "$KL" "$CONFIG_PATH"
  CONFIG_PATH="$CONFIG_PATH" LOG_LEVEL="$LOG_LEVEL" bash "$LAUNCH_SCRIPT"
done

printf '\nKL sweep completed. Generated configs in: %s\n' "$TMP_CONFIG_DIR"
