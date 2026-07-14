#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_deppa_full_pbvalid_lr1e5_wd1e4_b8x7_t200_100ep_20260714.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
MASTER_PORT="${MASTER_PORT:-29874}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
PB_ENV="${PB_ENV:-dif}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui-deppa-full-pbvalid-lr1e5-wd1e4-b8x7-t200-100ep_${RUN_STAMP}}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/logs/training.log}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/config" "$OUTPUT_DIR/launch_snapshot"
cp "$CONFIG_PATH" "$OUTPUT_DIR/config/launch_config.yaml"
cp "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launch_snapshot/$(basename "${BASH_SOURCE[0]}")"
CONFIG_SNAPSHOT="$OUTPUT_DIR/config/launch_config.yaml"

exec >>"$LOG_PATH" 2>&1

echo "[DiffGui-PBValid] start=$(date '+%F %T')"
echo "[DiffGui-PBValid] root=$ROOT"
echo "[DiffGui-PBValid] config=$CONFIG_SNAPSHOT"
echo "[DiffGui-PBValid] output=$OUTPUT_DIR"
echo "[DiffGui-PBValid] env=$CONDA_ENV pb_env=$PB_ENV"
echo "[DiffGui-PBValid] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[DiffGui-PBValid] nproc=$NPROC_PER_NODE batch_per_rank=8 global_batch=56 master_port=$MASTER_PORT"
echo "[DiffGui-PBValid] steps=200 epochs=100 ppo_epochs=1 checkpoint_every=1 seed_mode=gpu_id"
echo "[DiffGui-PBValid] logical_cpus_per_rank=4"

cd "$ROOT"
conda run -n "$PB_ENV" python -c "from posebusters import PoseBusters; PoseBusters(config='dock'); print('PoseBusters preflight OK')"
conda run -n "$CONDA_ENV" python -c "import torch, yaml; cfg=yaml.safe_load(open('$CONFIG_SNAPSHOT')); assert cfg['diffgui_training']['batch_size']==8; assert cfg['diffgui_training']['checkpoint_every_epochs']==1; assert cfg['diffgui_docking_batch_sweep']['num_timesteps']==200; assert cfg['diffgui_docking_batch_sweep']['seed_mode']=='gpu_id'; assert cfg['diffgui_posebusters_filter']['enabled'] is True; assert cfg['diffgui_runtime']['cpu_affinity']['logical_cpus_per_rank']==4; assert cfg['train']['ppo_epochs']==1; print('DiffGui config preflight OK', torch.__version__, torch.cuda.device_count())"

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  echo "[DiffGui-PBValid] PRECHECK_ONLY completed"
  exit 0
fi

conda run -n "$CONDA_ENV" torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "$CONFIG_SNAPSHOT" \
  --output-dir "$OUTPUT_DIR" \
  --log-level "$LOG_LEVEL"

echo "[DiffGui-PBValid] completed=$(date '+%F %T')"
