#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/md666/data/disk6/DiffB110"
BASE_CONFIG="$PROJECT_ROOT/configs/qed_sa_docking_8gjl_8gpu.yaml"
LAUNCH_SCRIPT="$PROJECT_ROOT/scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh"
TMP_CONFIG_DIR="$PROJECT_ROOT/configs/generated_kl_sweep"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-42}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
EPOCHS="${EPOCHS:-}"
KL_VALUES="${KL_VALUES:-0 0.01 0.02 0.05 0.1 0.2 0.5 1}"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "[ERROR] Base config not found: $BASE_CONFIG"
  exit 1
fi

if [ ! -f "$LAUNCH_SCRIPT" ]; then
  echo "[ERROR] Launch script not found: $LAUNCH_SCRIPT"
  exit 1
fi

mkdir -p "$TMP_CONFIG_DIR"

export PROJECT_ROOT BASE_CONFIG TMP_CONFIG_DIR SEED EPOCHS

printf '============================================================\n'
printf 'KL sweep: %s\n' "$KL_VALUES"
printf 'Base config : %s\n' "$BASE_CONFIG"
printf 'Launch script: %s\n' "$LAUNCH_SCRIPT"
printf 'GPUs        : %s\n' "$GPUS"
printf 'Seed        : %s\n' "$SEED"
printf 'Log level   : %s\n' "$LOG_LEVEL"
printf 'Epoch override: %s\n' "${EPOCHS:-<none>}"
printf 'Generated configs dir: %s\n' "$TMP_CONFIG_DIR"
printf '============================================================\n'

for KL in $KL_VALUES; do
  export KL
  KL_TAG=$(python - <<'PY'
import os
print(os.environ['KL'].replace('.', 'p'))
PY
)
  export KL_TAG

  CONFIG_PATH=$(python - <<'PY'
import os
from pathlib import Path
import yaml

base_config = Path(os.environ['BASE_CONFIG'])
out_dir = Path(os.environ['TMP_CONFIG_DIR'])
kl = float(os.environ['KL'])
kl_tag = os.environ['KL_TAG']
seed = int(os.environ['SEED'])
epochs_raw = os.environ.get('EPOCHS', '').strip()

data = yaml.safe_load(base_config.read_text(encoding='utf-8'))
run_name = data.get('name', 'qed-sa-docking-8gjl-8gpu')
data['name'] = f"{run_name}-kl-{kl_tag}"
data.setdefault('train', {})['kl_coef'] = kl
data['train']['seed'] = seed
if epochs_raw:
    data['train']['epochs'] = int(epochs_raw)

config_path = out_dir / f"qed_sa_docking_8gjl_8gpu_kl_{kl_tag}.yaml"
config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
print(str(config_path))
PY
)

  printf '\n============================================================\n'
  printf '[KL SWEEP] Starting run for kl_coef=%s\n' "$KL"
  printf '[KL SWEEP] Config: %s\n' "$CONFIG_PATH"
  printf '============================================================\n'

  CONFIG_PATH="$CONFIG_PATH" GPUS="$GPUS" SEED="$SEED" LOG_LEVEL="$LOG_LEVEL" bash "$LAUNCH_SCRIPT"
done

printf '\n============================================================\n'
printf 'KL sweep completed successfully\n'
printf 'Generated configs kept at: %s\n' "$TMP_CONFIG_DIR"
printf '============================================================\n'
